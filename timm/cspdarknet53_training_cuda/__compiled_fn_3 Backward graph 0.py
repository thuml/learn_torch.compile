from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[64]", primals_5: "f32[128]", primals_7: "f32[32]", primals_9: "f32[64]", primals_11: "f32[64]", primals_13: "f32[64]", primals_15: "f32[128]", primals_17: "f32[128]", primals_19: "f32[64]", primals_21: "f32[64]", primals_23: "f32[64]", primals_25: "f32[64]", primals_27: "f32[64]", primals_29: "f32[128]", primals_31: "f32[256]", primals_33: "f32[256]", primals_35: "f32[128]", primals_37: "f32[128]", primals_39: "f32[128]", primals_41: "f32[128]", primals_43: "f32[128]", primals_45: "f32[128]", primals_47: "f32[128]", primals_49: "f32[128]", primals_51: "f32[128]", primals_53: "f32[128]", primals_55: "f32[128]", primals_57: "f32[128]", primals_59: "f32[128]", primals_61: "f32[128]", primals_63: "f32[128]", primals_65: "f32[128]", primals_67: "f32[128]", primals_69: "f32[256]", primals_71: "f32[512]", primals_73: "f32[512]", primals_75: "f32[256]", primals_77: "f32[256]", primals_79: "f32[256]", primals_81: "f32[256]", primals_83: "f32[256]", primals_85: "f32[256]", primals_87: "f32[256]", primals_89: "f32[256]", primals_91: "f32[256]", primals_93: "f32[256]", primals_95: "f32[256]", primals_97: "f32[256]", primals_99: "f32[256]", primals_101: "f32[256]", primals_103: "f32[256]", primals_105: "f32[256]", primals_107: "f32[256]", primals_109: "f32[512]", primals_111: "f32[1024]", primals_113: "f32[1024]", primals_115: "f32[512]", primals_117: "f32[512]", primals_119: "f32[512]", primals_121: "f32[512]", primals_123: "f32[512]", primals_125: "f32[512]", primals_127: "f32[512]", primals_129: "f32[512]", primals_131: "f32[512]", primals_133: "f32[1024]", primals_135: "f32[32, 3, 3, 3]", primals_136: "f32[64, 32, 3, 3]", primals_137: "f32[128, 64, 1, 1]", primals_138: "f32[32, 64, 1, 1]", primals_139: "f32[64, 32, 3, 3]", primals_140: "f32[64, 64, 1, 1]", primals_141: "f32[64, 128, 1, 1]", primals_142: "f32[128, 64, 3, 3]", primals_143: "f32[128, 128, 1, 1]", primals_144: "f32[64, 64, 1, 1]", primals_145: "f32[64, 64, 3, 3]", primals_146: "f32[64, 64, 1, 1]", primals_147: "f32[64, 64, 3, 3]", primals_148: "f32[64, 64, 1, 1]", primals_149: "f32[128, 128, 1, 1]", primals_150: "f32[256, 128, 3, 3]", primals_151: "f32[256, 256, 1, 1]", primals_152: "f32[128, 128, 1, 1]", primals_153: "f32[128, 128, 3, 3]", primals_154: "f32[128, 128, 1, 1]", primals_155: "f32[128, 128, 3, 3]", primals_156: "f32[128, 128, 1, 1]", primals_157: "f32[128, 128, 3, 3]", primals_158: "f32[128, 128, 1, 1]", primals_159: "f32[128, 128, 3, 3]", primals_160: "f32[128, 128, 1, 1]", primals_161: "f32[128, 128, 3, 3]", primals_162: "f32[128, 128, 1, 1]", primals_163: "f32[128, 128, 3, 3]", primals_164: "f32[128, 128, 1, 1]", primals_165: "f32[128, 128, 3, 3]", primals_166: "f32[128, 128, 1, 1]", primals_167: "f32[128, 128, 3, 3]", primals_168: "f32[128, 128, 1, 1]", primals_169: "f32[256, 256, 1, 1]", primals_170: "f32[512, 256, 3, 3]", primals_171: "f32[512, 512, 1, 1]", primals_172: "f32[256, 256, 1, 1]", primals_173: "f32[256, 256, 3, 3]", primals_174: "f32[256, 256, 1, 1]", primals_175: "f32[256, 256, 3, 3]", primals_176: "f32[256, 256, 1, 1]", primals_177: "f32[256, 256, 3, 3]", primals_178: "f32[256, 256, 1, 1]", primals_179: "f32[256, 256, 3, 3]", primals_180: "f32[256, 256, 1, 1]", primals_181: "f32[256, 256, 3, 3]", primals_182: "f32[256, 256, 1, 1]", primals_183: "f32[256, 256, 3, 3]", primals_184: "f32[256, 256, 1, 1]", primals_185: "f32[256, 256, 3, 3]", primals_186: "f32[256, 256, 1, 1]", primals_187: "f32[256, 256, 3, 3]", primals_188: "f32[256, 256, 1, 1]", primals_189: "f32[512, 512, 1, 1]", primals_190: "f32[1024, 512, 3, 3]", primals_191: "f32[1024, 1024, 1, 1]", primals_192: "f32[512, 512, 1, 1]", primals_193: "f32[512, 512, 3, 3]", primals_194: "f32[512, 512, 1, 1]", primals_195: "f32[512, 512, 3, 3]", primals_196: "f32[512, 512, 1, 1]", primals_197: "f32[512, 512, 3, 3]", primals_198: "f32[512, 512, 1, 1]", primals_199: "f32[512, 512, 3, 3]", primals_200: "f32[512, 512, 1, 1]", primals_201: "f32[1024, 1024, 1, 1]", primals_405: "f32[8, 3, 256, 256]", convolution: "f32[8, 32, 256, 256]", squeeze_1: "f32[32]", where: "f32[8, 32, 256, 256]", convolution_1: "f32[8, 64, 128, 128]", squeeze_4: "f32[64]", where_1: "f32[8, 64, 128, 128]", convolution_2: "f32[8, 128, 128, 128]", squeeze_7: "f32[128]", getitem_9: "f32[8, 64, 128, 128]", convolution_3: "f32[8, 32, 128, 128]", squeeze_10: "f32[32]", where_3: "f32[8, 32, 128, 128]", convolution_4: "f32[8, 64, 128, 128]", squeeze_13: "f32[64]", add_25: "f32[8, 64, 128, 128]", convolution_5: "f32[8, 64, 128, 128]", squeeze_16: "f32[64]", cat: "f32[8, 128, 128, 128]", convolution_6: "f32[8, 64, 128, 128]", squeeze_19: "f32[64]", where_6: "f32[8, 64, 128, 128]", convolution_7: "f32[8, 128, 64, 64]", squeeze_22: "f32[128]", where_7: "f32[8, 128, 64, 64]", convolution_8: "f32[8, 128, 64, 64]", squeeze_25: "f32[128]", getitem_27: "f32[8, 64, 64, 64]", convolution_9: "f32[8, 64, 64, 64]", squeeze_28: "f32[64]", where_9: "f32[8, 64, 64, 64]", convolution_10: "f32[8, 64, 64, 64]", squeeze_31: "f32[64]", add_56: "f32[8, 64, 64, 64]", convolution_11: "f32[8, 64, 64, 64]", squeeze_34: "f32[64]", where_11: "f32[8, 64, 64, 64]", convolution_12: "f32[8, 64, 64, 64]", squeeze_37: "f32[64]", add_67: "f32[8, 64, 64, 64]", convolution_13: "f32[8, 64, 64, 64]", squeeze_40: "f32[64]", cat_1: "f32[8, 128, 64, 64]", convolution_14: "f32[8, 128, 64, 64]", squeeze_43: "f32[128]", where_14: "f32[8, 128, 64, 64]", convolution_15: "f32[8, 256, 32, 32]", squeeze_46: "f32[256]", where_15: "f32[8, 256, 32, 32]", convolution_16: "f32[8, 256, 32, 32]", squeeze_49: "f32[256]", getitem_49: "f32[8, 128, 32, 32]", convolution_17: "f32[8, 128, 32, 32]", squeeze_52: "f32[128]", where_17: "f32[8, 128, 32, 32]", convolution_18: "f32[8, 128, 32, 32]", squeeze_55: "f32[128]", add_98: "f32[8, 128, 32, 32]", convolution_19: "f32[8, 128, 32, 32]", squeeze_58: "f32[128]", where_19: "f32[8, 128, 32, 32]", convolution_20: "f32[8, 128, 32, 32]", squeeze_61: "f32[128]", add_109: "f32[8, 128, 32, 32]", convolution_21: "f32[8, 128, 32, 32]", squeeze_64: "f32[128]", where_21: "f32[8, 128, 32, 32]", convolution_22: "f32[8, 128, 32, 32]", squeeze_67: "f32[128]", add_120: "f32[8, 128, 32, 32]", convolution_23: "f32[8, 128, 32, 32]", squeeze_70: "f32[128]", where_23: "f32[8, 128, 32, 32]", convolution_24: "f32[8, 128, 32, 32]", squeeze_73: "f32[128]", add_131: "f32[8, 128, 32, 32]", convolution_25: "f32[8, 128, 32, 32]", squeeze_76: "f32[128]", where_25: "f32[8, 128, 32, 32]", convolution_26: "f32[8, 128, 32, 32]", squeeze_79: "f32[128]", add_142: "f32[8, 128, 32, 32]", convolution_27: "f32[8, 128, 32, 32]", squeeze_82: "f32[128]", where_27: "f32[8, 128, 32, 32]", convolution_28: "f32[8, 128, 32, 32]", squeeze_85: "f32[128]", add_153: "f32[8, 128, 32, 32]", convolution_29: "f32[8, 128, 32, 32]", squeeze_88: "f32[128]", where_29: "f32[8, 128, 32, 32]", convolution_30: "f32[8, 128, 32, 32]", squeeze_91: "f32[128]", add_164: "f32[8, 128, 32, 32]", convolution_31: "f32[8, 128, 32, 32]", squeeze_94: "f32[128]", where_31: "f32[8, 128, 32, 32]", convolution_32: "f32[8, 128, 32, 32]", squeeze_97: "f32[128]", add_175: "f32[8, 128, 32, 32]", convolution_33: "f32[8, 128, 32, 32]", squeeze_100: "f32[128]", cat_2: "f32[8, 256, 32, 32]", convolution_34: "f32[8, 256, 32, 32]", squeeze_103: "f32[256]", where_34: "f32[8, 256, 32, 32]", convolution_35: "f32[8, 512, 16, 16]", squeeze_106: "f32[512]", where_35: "f32[8, 512, 16, 16]", convolution_36: "f32[8, 512, 16, 16]", squeeze_109: "f32[512]", getitem_95: "f32[8, 256, 16, 16]", convolution_37: "f32[8, 256, 16, 16]", squeeze_112: "f32[256]", where_37: "f32[8, 256, 16, 16]", convolution_38: "f32[8, 256, 16, 16]", squeeze_115: "f32[256]", add_206: "f32[8, 256, 16, 16]", convolution_39: "f32[8, 256, 16, 16]", squeeze_118: "f32[256]", where_39: "f32[8, 256, 16, 16]", convolution_40: "f32[8, 256, 16, 16]", squeeze_121: "f32[256]", add_217: "f32[8, 256, 16, 16]", convolution_41: "f32[8, 256, 16, 16]", squeeze_124: "f32[256]", where_41: "f32[8, 256, 16, 16]", convolution_42: "f32[8, 256, 16, 16]", squeeze_127: "f32[256]", add_228: "f32[8, 256, 16, 16]", convolution_43: "f32[8, 256, 16, 16]", squeeze_130: "f32[256]", where_43: "f32[8, 256, 16, 16]", convolution_44: "f32[8, 256, 16, 16]", squeeze_133: "f32[256]", add_239: "f32[8, 256, 16, 16]", convolution_45: "f32[8, 256, 16, 16]", squeeze_136: "f32[256]", where_45: "f32[8, 256, 16, 16]", convolution_46: "f32[8, 256, 16, 16]", squeeze_139: "f32[256]", add_250: "f32[8, 256, 16, 16]", convolution_47: "f32[8, 256, 16, 16]", squeeze_142: "f32[256]", where_47: "f32[8, 256, 16, 16]", convolution_48: "f32[8, 256, 16, 16]", squeeze_145: "f32[256]", add_261: "f32[8, 256, 16, 16]", convolution_49: "f32[8, 256, 16, 16]", squeeze_148: "f32[256]", where_49: "f32[8, 256, 16, 16]", convolution_50: "f32[8, 256, 16, 16]", squeeze_151: "f32[256]", add_272: "f32[8, 256, 16, 16]", convolution_51: "f32[8, 256, 16, 16]", squeeze_154: "f32[256]", where_51: "f32[8, 256, 16, 16]", convolution_52: "f32[8, 256, 16, 16]", squeeze_157: "f32[256]", add_283: "f32[8, 256, 16, 16]", convolution_53: "f32[8, 256, 16, 16]", squeeze_160: "f32[256]", cat_3: "f32[8, 512, 16, 16]", convolution_54: "f32[8, 512, 16, 16]", squeeze_163: "f32[512]", where_54: "f32[8, 512, 16, 16]", convolution_55: "f32[8, 1024, 8, 8]", squeeze_166: "f32[1024]", where_55: "f32[8, 1024, 8, 8]", convolution_56: "f32[8, 1024, 8, 8]", squeeze_169: "f32[1024]", getitem_141: "f32[8, 512, 8, 8]", convolution_57: "f32[8, 512, 8, 8]", squeeze_172: "f32[512]", where_57: "f32[8, 512, 8, 8]", convolution_58: "f32[8, 512, 8, 8]", squeeze_175: "f32[512]", add_314: "f32[8, 512, 8, 8]", convolution_59: "f32[8, 512, 8, 8]", squeeze_178: "f32[512]", where_59: "f32[8, 512, 8, 8]", convolution_60: "f32[8, 512, 8, 8]", squeeze_181: "f32[512]", add_325: "f32[8, 512, 8, 8]", convolution_61: "f32[8, 512, 8, 8]", squeeze_184: "f32[512]", where_61: "f32[8, 512, 8, 8]", convolution_62: "f32[8, 512, 8, 8]", squeeze_187: "f32[512]", add_336: "f32[8, 512, 8, 8]", convolution_63: "f32[8, 512, 8, 8]", squeeze_190: "f32[512]", where_63: "f32[8, 512, 8, 8]", convolution_64: "f32[8, 512, 8, 8]", squeeze_193: "f32[512]", add_347: "f32[8, 512, 8, 8]", convolution_65: "f32[8, 512, 8, 8]", squeeze_196: "f32[512]", cat_4: "f32[8, 1024, 8, 8]", convolution_66: "f32[8, 1024, 8, 8]", squeeze_199: "f32[1024]", clone: "f32[8, 1024]", permute_1: "f32[1000, 1024]", gt_67: "b8[8, 1024, 8, 8]", unsqueeze_270: "f32[1, 1024, 1, 1]", gt_68: "b8[8, 512, 8, 8]", unsqueeze_282: "f32[1, 512, 1, 1]", gt_69: "b8[8, 512, 8, 8]", unsqueeze_294: "f32[1, 512, 1, 1]", unsqueeze_306: "f32[1, 512, 1, 1]", gt_71: "b8[8, 512, 8, 8]", unsqueeze_318: "f32[1, 512, 1, 1]", unsqueeze_330: "f32[1, 512, 1, 1]", gt_73: "b8[8, 512, 8, 8]", unsqueeze_342: "f32[1, 512, 1, 1]", unsqueeze_354: "f32[1, 512, 1, 1]", gt_75: "b8[8, 512, 8, 8]", unsqueeze_366: "f32[1, 512, 1, 1]", unsqueeze_378: "f32[1, 512, 1, 1]", gt_77: "b8[8, 1024, 8, 8]", unsqueeze_390: "f32[1, 1024, 1, 1]", unsqueeze_402: "f32[1, 1024, 1, 1]", unsqueeze_414: "f32[1, 512, 1, 1]", gt_80: "b8[8, 256, 16, 16]", unsqueeze_426: "f32[1, 256, 1, 1]", gt_81: "b8[8, 256, 16, 16]", unsqueeze_438: "f32[1, 256, 1, 1]", unsqueeze_450: "f32[1, 256, 1, 1]", gt_83: "b8[8, 256, 16, 16]", unsqueeze_462: "f32[1, 256, 1, 1]", unsqueeze_474: "f32[1, 256, 1, 1]", gt_85: "b8[8, 256, 16, 16]", unsqueeze_486: "f32[1, 256, 1, 1]", unsqueeze_498: "f32[1, 256, 1, 1]", gt_87: "b8[8, 256, 16, 16]", unsqueeze_510: "f32[1, 256, 1, 1]", unsqueeze_522: "f32[1, 256, 1, 1]", gt_89: "b8[8, 256, 16, 16]", unsqueeze_534: "f32[1, 256, 1, 1]", unsqueeze_546: "f32[1, 256, 1, 1]", gt_91: "b8[8, 256, 16, 16]", unsqueeze_558: "f32[1, 256, 1, 1]", unsqueeze_570: "f32[1, 256, 1, 1]", gt_93: "b8[8, 256, 16, 16]", unsqueeze_582: "f32[1, 256, 1, 1]", unsqueeze_594: "f32[1, 256, 1, 1]", gt_95: "b8[8, 256, 16, 16]", unsqueeze_606: "f32[1, 256, 1, 1]", unsqueeze_618: "f32[1, 256, 1, 1]", gt_97: "b8[8, 512, 16, 16]", unsqueeze_630: "f32[1, 512, 1, 1]", unsqueeze_642: "f32[1, 512, 1, 1]", unsqueeze_654: "f32[1, 256, 1, 1]", gt_100: "b8[8, 128, 32, 32]", unsqueeze_666: "f32[1, 128, 1, 1]", gt_101: "b8[8, 128, 32, 32]", unsqueeze_678: "f32[1, 128, 1, 1]", unsqueeze_690: "f32[1, 128, 1, 1]", gt_103: "b8[8, 128, 32, 32]", unsqueeze_702: "f32[1, 128, 1, 1]", unsqueeze_714: "f32[1, 128, 1, 1]", gt_105: "b8[8, 128, 32, 32]", unsqueeze_726: "f32[1, 128, 1, 1]", unsqueeze_738: "f32[1, 128, 1, 1]", gt_107: "b8[8, 128, 32, 32]", unsqueeze_750: "f32[1, 128, 1, 1]", unsqueeze_762: "f32[1, 128, 1, 1]", gt_109: "b8[8, 128, 32, 32]", unsqueeze_774: "f32[1, 128, 1, 1]", unsqueeze_786: "f32[1, 128, 1, 1]", gt_111: "b8[8, 128, 32, 32]", unsqueeze_798: "f32[1, 128, 1, 1]", unsqueeze_810: "f32[1, 128, 1, 1]", gt_113: "b8[8, 128, 32, 32]", unsqueeze_822: "f32[1, 128, 1, 1]", unsqueeze_834: "f32[1, 128, 1, 1]", gt_115: "b8[8, 128, 32, 32]", unsqueeze_846: "f32[1, 128, 1, 1]", unsqueeze_858: "f32[1, 128, 1, 1]", gt_117: "b8[8, 256, 32, 32]", unsqueeze_870: "f32[1, 256, 1, 1]", unsqueeze_882: "f32[1, 256, 1, 1]", unsqueeze_894: "f32[1, 128, 1, 1]", gt_120: "b8[8, 64, 64, 64]", unsqueeze_906: "f32[1, 64, 1, 1]", gt_121: "b8[8, 64, 64, 64]", unsqueeze_918: "f32[1, 64, 1, 1]", unsqueeze_930: "f32[1, 64, 1, 1]", gt_123: "b8[8, 64, 64, 64]", unsqueeze_942: "f32[1, 64, 1, 1]", unsqueeze_954: "f32[1, 64, 1, 1]", gt_125: "b8[8, 128, 64, 64]", unsqueeze_966: "f32[1, 128, 1, 1]", unsqueeze_978: "f32[1, 128, 1, 1]", unsqueeze_990: "f32[1, 64, 1, 1]", gt_128: "b8[8, 64, 128, 128]", unsqueeze_1002: "f32[1, 64, 1, 1]", gt_129: "b8[8, 64, 128, 128]", unsqueeze_1014: "f32[1, 64, 1, 1]", unsqueeze_1026: "f32[1, 32, 1, 1]", gt_131: "b8[8, 128, 128, 128]", unsqueeze_1038: "f32[1, 128, 1, 1]", unsqueeze_1050: "f32[1, 64, 1, 1]", unsqueeze_1062: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 8, 8]" = torch.ops.aten.expand.default(view_2, [8, 1024, 8, 8]);  view_2 = None
    div: "f32[8, 1024, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_536: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(div, 0.01)
    where_67: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_67, div, mul_536);  gt_67 = div = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_67, [0, 2, 3])
    sub_67: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_270);  convolution_66 = unsqueeze_270 = None
    mul_537: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_67, sub_67)
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
    mul_538: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_271: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
    unsqueeze_272: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_539: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_540: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, squeeze_199)
    mul_541: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
    unsqueeze_274: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_275: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_542: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_199, primals_133);  primals_133 = None
    unsqueeze_277: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_278: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_543: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_276);  sub_67 = unsqueeze_276 = None
    sub_69: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_67, mul_543);  where_67 = mul_543 = None
    sub_70: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_273);  sub_69 = unsqueeze_273 = None
    mul_544: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_279);  sub_70 = unsqueeze_279 = None
    mul_545: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_199);  sum_3 = squeeze_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_544, cat_4, primals_201, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_544 = cat_4 = primals_201 = None
    getitem_164: "f32[8, 1024, 8, 8]" = convolution_backward[0]
    getitem_165: "f32[1024, 1024, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_1: "f32[8, 512, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_164, 1, 0, 512)
    slice_2: "f32[8, 512, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_164, 1, 512, 1024);  getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_546: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(slice_2, 0.01)
    where_68: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_68, slice_2, mul_546);  gt_68 = slice_2 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_71: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_282);  convolution_65 = unsqueeze_282 = None
    mul_547: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_68, sub_71)
    sum_5: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[512]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_283: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_284: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_549: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_550: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, squeeze_196)
    mul_551: "f32[512]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_286: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_287: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_552: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_196, primals_131);  primals_131 = None
    unsqueeze_289: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_290: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_553: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_288);  sub_71 = unsqueeze_288 = None
    sub_73: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_68, mul_553);  where_68 = mul_553 = None
    sub_74: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_285);  sub_73 = unsqueeze_285 = None
    mul_554: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_291);  sub_74 = unsqueeze_291 = None
    mul_555: "f32[512]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_196);  sum_5 = squeeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_554, add_347, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = add_347 = primals_200 = None
    getitem_167: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_168: "f32[512, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_556: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_167, 0.01)
    where_69: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_69, getitem_167, mul_556);  gt_69 = mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_69, [0, 2, 3])
    sub_75: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_294);  convolution_64 = unsqueeze_294 = None
    mul_557: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_69, sub_75)
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_558: "f32[512]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_295: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_296: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_559: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_560: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_561: "f32[512]" = torch.ops.aten.mul.Tensor(mul_559, mul_560);  mul_559 = mul_560 = None
    unsqueeze_298: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_299: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_562: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_301: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_302: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_563: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_300);  sub_75 = unsqueeze_300 = None
    sub_77: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_69, mul_563);  where_69 = mul_563 = None
    sub_78: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_297);  sub_77 = unsqueeze_297 = None
    mul_564: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_303);  sub_78 = unsqueeze_303 = None
    mul_565: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_193);  sum_7 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_564, where_63, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_564 = primals_199 = None
    getitem_170: "f32[8, 512, 8, 8]" = convolution_backward_2[0]
    getitem_171: "f32[512, 512, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_77: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_63);  where_63 = None
    alias_78: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    gt_70: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_78, 0);  alias_78 = None
    mul_566: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_170, 0.01)
    where_70: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_70, getitem_170, mul_566);  gt_70 = getitem_170 = mul_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_70, [0, 2, 3])
    sub_79: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_306);  convolution_63 = unsqueeze_306 = None
    mul_567: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_70, sub_79)
    sum_9: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 2, 3]);  mul_567 = None
    mul_568: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_307: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_308: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_569: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_570: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_571: "f32[512]" = torch.ops.aten.mul.Tensor(mul_569, mul_570);  mul_569 = mul_570 = None
    unsqueeze_310: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_311: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_572: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_313: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_314: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_573: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_312);  sub_79 = unsqueeze_312 = None
    sub_81: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_70, mul_573);  where_70 = mul_573 = None
    sub_82: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_309);  sub_81 = unsqueeze_309 = None
    mul_574: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_315);  sub_82 = unsqueeze_315 = None
    mul_575: "f32[512]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_190);  sum_9 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_574, add_336, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_574 = add_336 = primals_198 = None
    getitem_173: "f32[8, 512, 8, 8]" = convolution_backward_3[0]
    getitem_174: "f32[512, 512, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_358: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(getitem_167, getitem_173);  getitem_167 = getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_576: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_358, 0.01)
    where_71: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_71, add_358, mul_576);  gt_71 = mul_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_71, [0, 2, 3])
    sub_83: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_318);  convolution_62 = unsqueeze_318 = None
    mul_577: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_71, sub_83)
    sum_11: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3]);  mul_577 = None
    mul_578: "f32[512]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_319: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
    unsqueeze_320: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_579: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_580: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_581: "f32[512]" = torch.ops.aten.mul.Tensor(mul_579, mul_580);  mul_579 = mul_580 = None
    unsqueeze_322: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_323: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_582: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_325: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_326: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_583: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_324);  sub_83 = unsqueeze_324 = None
    sub_85: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_71, mul_583);  where_71 = mul_583 = None
    sub_86: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_321);  sub_85 = unsqueeze_321 = None
    mul_584: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_327);  sub_86 = unsqueeze_327 = None
    mul_585: "f32[512]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_187);  sum_11 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_584, where_61, primals_197, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_584 = primals_197 = None
    getitem_176: "f32[8, 512, 8, 8]" = convolution_backward_4[0]
    getitem_177: "f32[512, 512, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_83: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_61);  where_61 = None
    alias_84: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    gt_72: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_84, 0);  alias_84 = None
    mul_586: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_176, 0.01)
    where_72: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_72, getitem_176, mul_586);  gt_72 = getitem_176 = mul_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_72, [0, 2, 3])
    sub_87: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330);  convolution_61 = unsqueeze_330 = None
    mul_587: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_72, sub_87)
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
    mul_588: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_331: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_332: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_589: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_590: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_591: "f32[512]" = torch.ops.aten.mul.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
    unsqueeze_334: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_335: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_592: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_337: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_338: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_593: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_336);  sub_87 = unsqueeze_336 = None
    sub_89: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_72, mul_593);  where_72 = mul_593 = None
    sub_90: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_333);  sub_89 = unsqueeze_333 = None
    mul_594: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_339);  sub_90 = unsqueeze_339 = None
    mul_595: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_184);  sum_13 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_594, add_325, primals_196, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = add_325 = primals_196 = None
    getitem_179: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_180: "f32[512, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_359: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_358, getitem_179);  add_358 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_596: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_359, 0.01)
    where_73: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_73, add_359, mul_596);  gt_73 = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_73, [0, 2, 3])
    sub_91: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342);  convolution_60 = unsqueeze_342 = None
    mul_597: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_73, sub_91)
    sum_15: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_598: "f32[512]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_343: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_344: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_599: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_600: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_601: "f32[512]" = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_346: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_347: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_602: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_349: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_350: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_603: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_348);  sub_91 = unsqueeze_348 = None
    sub_93: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_73, mul_603);  where_73 = mul_603 = None
    sub_94: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_345);  sub_93 = unsqueeze_345 = None
    mul_604: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_351);  sub_94 = unsqueeze_351 = None
    mul_605: "f32[512]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_181);  sum_15 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_604, where_59, primals_195, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_604 = primals_195 = None
    getitem_182: "f32[8, 512, 8, 8]" = convolution_backward_6[0]
    getitem_183: "f32[512, 512, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_89: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_59);  where_59 = None
    alias_90: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    gt_74: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_90, 0);  alias_90 = None
    mul_606: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_182, 0.01)
    where_74: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_74, getitem_182, mul_606);  gt_74 = getitem_182 = mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_74, [0, 2, 3])
    sub_95: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354);  convolution_59 = unsqueeze_354 = None
    mul_607: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_74, sub_95)
    sum_17: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[512]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_355: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_356: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_609: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_610: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_611: "f32[512]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_358: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_359: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_612: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_361: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_362: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_613: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_360);  sub_95 = unsqueeze_360 = None
    sub_97: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_74, mul_613);  where_74 = mul_613 = None
    sub_98: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_357);  sub_97 = unsqueeze_357 = None
    mul_614: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_363);  sub_98 = unsqueeze_363 = None
    mul_615: "f32[512]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_178);  sum_17 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_614, add_314, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = add_314 = primals_194 = None
    getitem_185: "f32[8, 512, 8, 8]" = convolution_backward_7[0]
    getitem_186: "f32[512, 512, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_360: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_359, getitem_185);  add_359 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_616: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(add_360, 0.01)
    where_75: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_75, add_360, mul_616);  gt_75 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_75, [0, 2, 3])
    sub_99: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_366);  convolution_58 = unsqueeze_366 = None
    mul_617: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_75, sub_99)
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
    mul_618: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_367: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_368: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_619: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_620: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_621: "f32[512]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_370: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_371: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_622: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_373: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_374: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_623: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_372);  sub_99 = unsqueeze_372 = None
    sub_101: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_75, mul_623);  where_75 = mul_623 = None
    sub_102: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_369);  sub_101 = unsqueeze_369 = None
    mul_624: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_375);  sub_102 = unsqueeze_375 = None
    mul_625: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_175);  sum_19 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_624, where_57, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = primals_193 = None
    getitem_188: "f32[8, 512, 8, 8]" = convolution_backward_8[0]
    getitem_189: "f32[512, 512, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_95: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(where_57);  where_57 = None
    alias_96: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    gt_76: "b8[8, 512, 8, 8]" = torch.ops.aten.gt.Scalar(alias_96, 0);  alias_96 = None
    mul_626: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_188, 0.01)
    where_76: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(gt_76, getitem_188, mul_626);  gt_76 = getitem_188 = mul_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_76, [0, 2, 3])
    sub_103: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_378);  convolution_57 = unsqueeze_378 = None
    mul_627: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_76, sub_103)
    sum_21: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_628: "f32[512]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    unsqueeze_379: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_380: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_629: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    mul_630: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_631: "f32[512]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_382: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_383: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_632: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_385: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_386: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_633: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_384);  sub_103 = unsqueeze_384 = None
    sub_105: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_76, mul_633);  where_76 = mul_633 = None
    sub_106: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_381);  sub_105 = unsqueeze_381 = None
    mul_634: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_387);  sub_106 = unsqueeze_387 = None
    mul_635: "f32[512]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_172);  sum_21 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_634, getitem_141, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_634 = getitem_141 = primals_192 = None
    getitem_191: "f32[8, 512, 8, 8]" = convolution_backward_9[0]
    getitem_192: "f32[512, 512, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_361: "f32[8, 512, 8, 8]" = torch.ops.aten.add.Tensor(add_360, getitem_191);  add_360 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_5: "f32[8, 1024, 8, 8]" = torch.ops.aten.cat.default([slice_1, add_361], 1);  slice_1 = add_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_636: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(cat_5, 0.01)
    where_77: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_77, cat_5, mul_636);  gt_77 = cat_5 = mul_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_77, [0, 2, 3])
    sub_107: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_390);  convolution_56 = unsqueeze_390 = None
    mul_637: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_77, sub_107)
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 2, 3]);  mul_637 = None
    mul_638: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    unsqueeze_391: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_392: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_639: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, 0.001953125)
    mul_640: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_641: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
    unsqueeze_394: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_395: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_642: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_397: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_398: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_643: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_396);  sub_107 = unsqueeze_396 = None
    sub_109: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_77, mul_643);  where_77 = mul_643 = None
    sub_110: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_393);  sub_109 = unsqueeze_393 = None
    mul_644: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_399);  sub_110 = unsqueeze_399 = None
    mul_645: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_169);  sum_23 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_644, where_55, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_644 = primals_191 = None
    getitem_194: "f32[8, 1024, 8, 8]" = convolution_backward_10[0]
    getitem_195: "f32[1024, 1024, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_101: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(where_55);  where_55 = None
    alias_102: "f32[8, 1024, 8, 8]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    gt_78: "b8[8, 1024, 8, 8]" = torch.ops.aten.gt.Scalar(alias_102, 0);  alias_102 = None
    mul_646: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_194, 0.01)
    where_78: "f32[8, 1024, 8, 8]" = torch.ops.aten.where.self(gt_78, getitem_194, mul_646);  gt_78 = getitem_194 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_78, [0, 2, 3])
    sub_111: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_402);  convolution_55 = unsqueeze_402 = None
    mul_647: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(where_78, sub_111)
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3]);  mul_647 = None
    mul_648: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_24, 0.001953125)
    unsqueeze_403: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_404: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_649: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.001953125)
    mul_650: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_651: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
    unsqueeze_406: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_407: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_652: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_409: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_410: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_653: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_408);  sub_111 = unsqueeze_408 = None
    sub_113: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(where_78, mul_653);  where_78 = mul_653 = None
    sub_114: "f32[8, 1024, 8, 8]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_405);  sub_113 = unsqueeze_405 = None
    mul_654: "f32[8, 1024, 8, 8]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_411);  sub_114 = unsqueeze_411 = None
    mul_655: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_166);  sum_25 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_654, where_54, primals_190, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_654 = primals_190 = None
    getitem_197: "f32[8, 512, 16, 16]" = convolution_backward_11[0]
    getitem_198: "f32[1024, 512, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_104: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_54);  where_54 = None
    alias_105: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    gt_79: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_105, 0);  alias_105 = None
    mul_656: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_197, 0.01)
    where_79: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_79, getitem_197, mul_656);  gt_79 = getitem_197 = mul_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_79, [0, 2, 3])
    sub_115: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_414);  convolution_54 = unsqueeze_414 = None
    mul_657: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_79, sub_115)
    sum_27: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 2, 3]);  mul_657 = None
    mul_658: "f32[512]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    unsqueeze_415: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_416: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_659: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    mul_660: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_661: "f32[512]" = torch.ops.aten.mul.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
    unsqueeze_418: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_419: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_662: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_421: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_422: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_663: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_420);  sub_115 = unsqueeze_420 = None
    sub_117: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_79, mul_663);  where_79 = mul_663 = None
    sub_118: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_417);  sub_117 = unsqueeze_417 = None
    mul_664: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_423);  sub_118 = unsqueeze_423 = None
    mul_665: "f32[512]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_163);  sum_27 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_664, cat_3, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_664 = cat_3 = primals_189 = None
    getitem_200: "f32[8, 512, 16, 16]" = convolution_backward_12[0]
    getitem_201: "f32[512, 512, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_3: "f32[8, 256, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_200, 1, 0, 256)
    slice_4: "f32[8, 256, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_200, 1, 256, 512);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_666: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(slice_4, 0.01)
    where_80: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_80, slice_4, mul_666);  gt_80 = slice_4 = mul_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_80, [0, 2, 3])
    sub_119: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_426);  convolution_53 = unsqueeze_426 = None
    mul_667: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_80, sub_119)
    sum_29: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_667, [0, 2, 3]);  mul_667 = None
    mul_668: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    unsqueeze_427: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_428: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_669: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, 0.00048828125)
    mul_670: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_671: "f32[256]" = torch.ops.aten.mul.Tensor(mul_669, mul_670);  mul_669 = mul_670 = None
    unsqueeze_430: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_431: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_672: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_433: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_434: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_673: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_432);  sub_119 = unsqueeze_432 = None
    sub_121: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_80, mul_673);  where_80 = mul_673 = None
    sub_122: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_429);  sub_121 = unsqueeze_429 = None
    mul_674: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_435);  sub_122 = unsqueeze_435 = None
    mul_675: "f32[256]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_160);  sum_29 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_674, add_283, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_674 = add_283 = primals_188 = None
    getitem_203: "f32[8, 256, 16, 16]" = convolution_backward_13[0]
    getitem_204: "f32[256, 256, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_676: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_203, 0.01)
    where_81: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_81, getitem_203, mul_676);  gt_81 = mul_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_81, [0, 2, 3])
    sub_123: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_438);  convolution_52 = unsqueeze_438 = None
    mul_677: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_81, sub_123)
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_677, [0, 2, 3]);  mul_677 = None
    mul_678: "f32[256]" = torch.ops.aten.mul.Tensor(sum_30, 0.00048828125)
    unsqueeze_439: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_440: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_679: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, 0.00048828125)
    mul_680: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_681: "f32[256]" = torch.ops.aten.mul.Tensor(mul_679, mul_680);  mul_679 = mul_680 = None
    unsqueeze_442: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_443: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_682: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_445: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_446: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_683: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_444);  sub_123 = unsqueeze_444 = None
    sub_125: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_81, mul_683);  where_81 = mul_683 = None
    sub_126: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_441);  sub_125 = unsqueeze_441 = None
    mul_684: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_447);  sub_126 = unsqueeze_447 = None
    mul_685: "f32[256]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_157);  sum_31 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_684, where_51, primals_187, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_684 = primals_187 = None
    getitem_206: "f32[8, 256, 16, 16]" = convolution_backward_14[0]
    getitem_207: "f32[256, 256, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_113: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_51);  where_51 = None
    alias_114: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    gt_82: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_114, 0);  alias_114 = None
    mul_686: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_206, 0.01)
    where_82: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_82, getitem_206, mul_686);  gt_82 = getitem_206 = mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_82, [0, 2, 3])
    sub_127: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_450);  convolution_51 = unsqueeze_450 = None
    mul_687: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_82, sub_127)
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 2, 3]);  mul_687 = None
    mul_688: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_451: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_452: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_690: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_691: "f32[256]" = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
    unsqueeze_454: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_455: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_692: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_457: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_458: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_693: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_456);  sub_127 = unsqueeze_456 = None
    sub_129: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_82, mul_693);  where_82 = mul_693 = None
    sub_130: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_453);  sub_129 = unsqueeze_453 = None
    mul_694: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_459);  sub_130 = unsqueeze_459 = None
    mul_695: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_154);  sum_33 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_694, add_272, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_694 = add_272 = primals_186 = None
    getitem_209: "f32[8, 256, 16, 16]" = convolution_backward_15[0]
    getitem_210: "f32[256, 256, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_362: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(getitem_203, getitem_209);  getitem_203 = getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_696: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_362, 0.01)
    where_83: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_83, add_362, mul_696);  gt_83 = mul_696 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_83, [0, 2, 3])
    sub_131: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_462);  convolution_50 = unsqueeze_462 = None
    mul_697: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_83, sub_131)
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_698: "f32[256]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_463: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_464: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_699: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_700: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_701: "f32[256]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_466: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_467: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_702: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_469: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_470: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_703: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_468);  sub_131 = unsqueeze_468 = None
    sub_133: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_83, mul_703);  where_83 = mul_703 = None
    sub_134: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_465);  sub_133 = unsqueeze_465 = None
    mul_704: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_471);  sub_134 = unsqueeze_471 = None
    mul_705: "f32[256]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_151);  sum_35 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_704, where_49, primals_185, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_185 = None
    getitem_212: "f32[8, 256, 16, 16]" = convolution_backward_16[0]
    getitem_213: "f32[256, 256, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_119: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_49);  where_49 = None
    alias_120: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    gt_84: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_120, 0);  alias_120 = None
    mul_706: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_212, 0.01)
    where_84: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_84, getitem_212, mul_706);  gt_84 = getitem_212 = mul_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_84, [0, 2, 3])
    sub_135: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_474);  convolution_49 = unsqueeze_474 = None
    mul_707: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_84, sub_135)
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(sum_36, 0.00048828125)
    unsqueeze_475: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_476: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_711: "f32[256]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_478: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_479: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_712: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_481: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_482: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_713: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_480);  sub_135 = unsqueeze_480 = None
    sub_137: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_84, mul_713);  where_84 = mul_713 = None
    sub_138: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_477);  sub_137 = unsqueeze_477 = None
    mul_714: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_483);  sub_138 = unsqueeze_483 = None
    mul_715: "f32[256]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_148);  sum_37 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_714, add_261, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = add_261 = primals_184 = None
    getitem_215: "f32[8, 256, 16, 16]" = convolution_backward_17[0]
    getitem_216: "f32[256, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_363: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_362, getitem_215);  add_362 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_716: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_363, 0.01)
    where_85: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_85, add_363, mul_716);  gt_85 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_85, [0, 2, 3])
    sub_139: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_486);  convolution_48 = unsqueeze_486 = None
    mul_717: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_85, sub_139)
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    unsqueeze_487: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_488: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_490: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_491: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_493: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_494: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_723: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_492);  sub_139 = unsqueeze_492 = None
    sub_141: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_85, mul_723);  where_85 = mul_723 = None
    sub_142: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_489);  sub_141 = unsqueeze_489 = None
    mul_724: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_495);  sub_142 = unsqueeze_495 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_145);  sum_39 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_724, where_47, primals_183, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = primals_183 = None
    getitem_218: "f32[8, 256, 16, 16]" = convolution_backward_18[0]
    getitem_219: "f32[256, 256, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_125: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_47);  where_47 = None
    alias_126: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    gt_86: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_126, 0);  alias_126 = None
    mul_726: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_218, 0.01)
    where_86: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_86, getitem_218, mul_726);  gt_86 = getitem_218 = mul_726 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_86, [0, 2, 3])
    sub_143: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_498);  convolution_47 = unsqueeze_498 = None
    mul_727: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_86, sub_143)
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 2, 3]);  mul_727 = None
    mul_728: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.00048828125)
    unsqueeze_499: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_500: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_729: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.00048828125)
    mul_730: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_731: "f32[256]" = torch.ops.aten.mul.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    unsqueeze_502: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_503: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_732: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_505: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_506: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_733: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_504);  sub_143 = unsqueeze_504 = None
    sub_145: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_86, mul_733);  where_86 = mul_733 = None
    sub_146: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_501);  sub_145 = unsqueeze_501 = None
    mul_734: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_507);  sub_146 = unsqueeze_507 = None
    mul_735: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_142);  sum_41 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_734, add_250, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_734 = add_250 = primals_182 = None
    getitem_221: "f32[8, 256, 16, 16]" = convolution_backward_19[0]
    getitem_222: "f32[256, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_364: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_363, getitem_221);  add_363 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_736: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_364, 0.01)
    where_87: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_87, add_364, mul_736);  gt_87 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_87, [0, 2, 3])
    sub_147: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_510);  convolution_46 = unsqueeze_510 = None
    mul_737: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_87, sub_147)
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
    mul_738: "f32[256]" = torch.ops.aten.mul.Tensor(sum_42, 0.00048828125)
    unsqueeze_511: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_512: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_739: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, 0.00048828125)
    mul_740: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
    unsqueeze_514: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_515: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_742: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_517: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_518: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_743: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_516);  sub_147 = unsqueeze_516 = None
    sub_149: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_87, mul_743);  where_87 = mul_743 = None
    sub_150: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_513);  sub_149 = unsqueeze_513 = None
    mul_744: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_519);  sub_150 = unsqueeze_519 = None
    mul_745: "f32[256]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_139);  sum_43 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_744, where_45, primals_181, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_744 = primals_181 = None
    getitem_224: "f32[8, 256, 16, 16]" = convolution_backward_20[0]
    getitem_225: "f32[256, 256, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_131: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_45);  where_45 = None
    alias_132: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    gt_88: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_132, 0);  alias_132 = None
    mul_746: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_224, 0.01)
    where_88: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_88, getitem_224, mul_746);  gt_88 = getitem_224 = mul_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_88, [0, 2, 3])
    sub_151: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_522);  convolution_45 = unsqueeze_522 = None
    mul_747: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_88, sub_151)
    sum_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 2, 3]);  mul_747 = None
    mul_748: "f32[256]" = torch.ops.aten.mul.Tensor(sum_44, 0.00048828125)
    unsqueeze_523: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_524: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_749: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, 0.00048828125)
    mul_750: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_751: "f32[256]" = torch.ops.aten.mul.Tensor(mul_749, mul_750);  mul_749 = mul_750 = None
    unsqueeze_526: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_527: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_529: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_530: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_753: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_528);  sub_151 = unsqueeze_528 = None
    sub_153: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_88, mul_753);  where_88 = mul_753 = None
    sub_154: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_525);  sub_153 = unsqueeze_525 = None
    mul_754: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_531);  sub_154 = unsqueeze_531 = None
    mul_755: "f32[256]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_136);  sum_45 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_754, add_239, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_754 = add_239 = primals_180 = None
    getitem_227: "f32[8, 256, 16, 16]" = convolution_backward_21[0]
    getitem_228: "f32[256, 256, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_365: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_364, getitem_227);  add_364 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_756: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_365, 0.01)
    where_89: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_89, add_365, mul_756);  gt_89 = mul_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_89, [0, 2, 3])
    sub_155: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_534);  convolution_44 = unsqueeze_534 = None
    mul_757: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_89, sub_155)
    sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 2, 3]);  mul_757 = None
    mul_758: "f32[256]" = torch.ops.aten.mul.Tensor(sum_46, 0.00048828125)
    unsqueeze_535: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_536: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_759: "f32[256]" = torch.ops.aten.mul.Tensor(sum_47, 0.00048828125)
    mul_760: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_761: "f32[256]" = torch.ops.aten.mul.Tensor(mul_759, mul_760);  mul_759 = mul_760 = None
    unsqueeze_538: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_539: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_762: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_541: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_542: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_763: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_540);  sub_155 = unsqueeze_540 = None
    sub_157: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_89, mul_763);  where_89 = mul_763 = None
    sub_158: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_537);  sub_157 = unsqueeze_537 = None
    mul_764: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_543);  sub_158 = unsqueeze_543 = None
    mul_765: "f32[256]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_133);  sum_47 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_764, where_43, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_764 = primals_179 = None
    getitem_230: "f32[8, 256, 16, 16]" = convolution_backward_22[0]
    getitem_231: "f32[256, 256, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_137: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_43);  where_43 = None
    alias_138: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    gt_90: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_138, 0);  alias_138 = None
    mul_766: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_230, 0.01)
    where_90: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_90, getitem_230, mul_766);  gt_90 = getitem_230 = mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_90, [0, 2, 3])
    sub_159: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_546);  convolution_43 = unsqueeze_546 = None
    mul_767: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_90, sub_159)
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3]);  mul_767 = None
    mul_768: "f32[256]" = torch.ops.aten.mul.Tensor(sum_48, 0.00048828125)
    unsqueeze_547: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_548: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_769: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, 0.00048828125)
    mul_770: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_771: "f32[256]" = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
    unsqueeze_550: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_551: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_772: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_553: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_554: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_773: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_552);  sub_159 = unsqueeze_552 = None
    sub_161: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_90, mul_773);  where_90 = mul_773 = None
    sub_162: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_549);  sub_161 = unsqueeze_549 = None
    mul_774: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_555);  sub_162 = unsqueeze_555 = None
    mul_775: "f32[256]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_130);  sum_49 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_774, add_228, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_774 = add_228 = primals_178 = None
    getitem_233: "f32[8, 256, 16, 16]" = convolution_backward_23[0]
    getitem_234: "f32[256, 256, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_366: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_365, getitem_233);  add_365 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_776: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_366, 0.01)
    where_91: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_91, add_366, mul_776);  gt_91 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_91, [0, 2, 3])
    sub_163: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_558);  convolution_42 = unsqueeze_558 = None
    mul_777: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_91, sub_163)
    sum_51: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
    mul_778: "f32[256]" = torch.ops.aten.mul.Tensor(sum_50, 0.00048828125)
    unsqueeze_559: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_560: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_779: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, 0.00048828125)
    mul_780: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_781: "f32[256]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
    unsqueeze_562: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_563: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_782: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_565: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_566: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_783: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_564);  sub_163 = unsqueeze_564 = None
    sub_165: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_91, mul_783);  where_91 = mul_783 = None
    sub_166: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_561);  sub_165 = unsqueeze_561 = None
    mul_784: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_567);  sub_166 = unsqueeze_567 = None
    mul_785: "f32[256]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_127);  sum_51 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_784, where_41, primals_177, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = primals_177 = None
    getitem_236: "f32[8, 256, 16, 16]" = convolution_backward_24[0]
    getitem_237: "f32[256, 256, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_143: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_41);  where_41 = None
    alias_144: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    gt_92: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_144, 0);  alias_144 = None
    mul_786: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_236, 0.01)
    where_92: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_92, getitem_236, mul_786);  gt_92 = getitem_236 = mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_92, [0, 2, 3])
    sub_167: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_570);  convolution_41 = unsqueeze_570 = None
    mul_787: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_92, sub_167)
    sum_53: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_788: "f32[256]" = torch.ops.aten.mul.Tensor(sum_52, 0.00048828125)
    unsqueeze_571: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_572: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_789: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, 0.00048828125)
    mul_790: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_791: "f32[256]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
    unsqueeze_574: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_575: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_792: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_577: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_578: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_793: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_576);  sub_167 = unsqueeze_576 = None
    sub_169: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_92, mul_793);  where_92 = mul_793 = None
    sub_170: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_573);  sub_169 = unsqueeze_573 = None
    mul_794: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_579);  sub_170 = unsqueeze_579 = None
    mul_795: "f32[256]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_124);  sum_53 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_794, add_217, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_794 = add_217 = primals_176 = None
    getitem_239: "f32[8, 256, 16, 16]" = convolution_backward_25[0]
    getitem_240: "f32[256, 256, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_367: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_366, getitem_239);  add_366 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_796: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_367, 0.01)
    where_93: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_93, add_367, mul_796);  gt_93 = mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_93, [0, 2, 3])
    sub_171: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_582);  convolution_40 = unsqueeze_582 = None
    mul_797: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_93, sub_171)
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
    mul_798: "f32[256]" = torch.ops.aten.mul.Tensor(sum_54, 0.00048828125)
    unsqueeze_583: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_584: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_799: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, 0.00048828125)
    mul_800: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_801: "f32[256]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    unsqueeze_586: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_587: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_802: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_589: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_590: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_803: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_588);  sub_171 = unsqueeze_588 = None
    sub_173: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_93, mul_803);  where_93 = mul_803 = None
    sub_174: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_585);  sub_173 = unsqueeze_585 = None
    mul_804: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_591);  sub_174 = unsqueeze_591 = None
    mul_805: "f32[256]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_121);  sum_55 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_804, where_39, primals_175, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_804 = primals_175 = None
    getitem_242: "f32[8, 256, 16, 16]" = convolution_backward_26[0]
    getitem_243: "f32[256, 256, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_149: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_39);  where_39 = None
    alias_150: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    gt_94: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_150, 0);  alias_150 = None
    mul_806: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_242, 0.01)
    where_94: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_94, getitem_242, mul_806);  gt_94 = getitem_242 = mul_806 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_94, [0, 2, 3])
    sub_175: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_594);  convolution_39 = unsqueeze_594 = None
    mul_807: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_94, sub_175)
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
    mul_808: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 0.00048828125)
    unsqueeze_595: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_596: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_809: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 0.00048828125)
    mul_810: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_811: "f32[256]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
    unsqueeze_598: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_599: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_812: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_601: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_602: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_813: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_600);  sub_175 = unsqueeze_600 = None
    sub_177: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_94, mul_813);  where_94 = mul_813 = None
    sub_178: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_597);  sub_177 = unsqueeze_597 = None
    mul_814: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_603);  sub_178 = unsqueeze_603 = None
    mul_815: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_118);  sum_57 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_814, add_206, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = add_206 = primals_174 = None
    getitem_245: "f32[8, 256, 16, 16]" = convolution_backward_27[0]
    getitem_246: "f32[256, 256, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_368: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_367, getitem_245);  add_367 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_816: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(add_368, 0.01)
    where_95: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_95, add_368, mul_816);  gt_95 = mul_816 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_95, [0, 2, 3])
    sub_179: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_606);  convolution_38 = unsqueeze_606 = None
    mul_817: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_95, sub_179)
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3]);  mul_817 = None
    mul_818: "f32[256]" = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
    unsqueeze_607: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_608: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_819: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
    mul_820: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_821: "f32[256]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_610: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_611: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_822: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_613: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_614: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_823: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_612);  sub_179 = unsqueeze_612 = None
    sub_181: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_95, mul_823);  where_95 = mul_823 = None
    sub_182: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_609);  sub_181 = unsqueeze_609 = None
    mul_824: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_615);  sub_182 = unsqueeze_615 = None
    mul_825: "f32[256]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_115);  sum_59 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_824, where_37, primals_173, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_824 = primals_173 = None
    getitem_248: "f32[8, 256, 16, 16]" = convolution_backward_28[0]
    getitem_249: "f32[256, 256, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_155: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(where_37);  where_37 = None
    alias_156: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    gt_96: "b8[8, 256, 16, 16]" = torch.ops.aten.gt.Scalar(alias_156, 0);  alias_156 = None
    mul_826: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_248, 0.01)
    where_96: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(gt_96, getitem_248, mul_826);  gt_96 = getitem_248 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_96, [0, 2, 3])
    sub_183: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_618);  convolution_37 = unsqueeze_618 = None
    mul_827: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_96, sub_183)
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3]);  mul_827 = None
    mul_828: "f32[256]" = torch.ops.aten.mul.Tensor(sum_60, 0.00048828125)
    unsqueeze_619: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_620: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_829: "f32[256]" = torch.ops.aten.mul.Tensor(sum_61, 0.00048828125)
    mul_830: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_831: "f32[256]" = torch.ops.aten.mul.Tensor(mul_829, mul_830);  mul_829 = mul_830 = None
    unsqueeze_622: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_623: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_832: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_625: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_626: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_833: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_624);  sub_183 = unsqueeze_624 = None
    sub_185: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_96, mul_833);  where_96 = mul_833 = None
    sub_186: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_621);  sub_185 = unsqueeze_621 = None
    mul_834: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_627);  sub_186 = unsqueeze_627 = None
    mul_835: "f32[256]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_112);  sum_61 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_834, getitem_95, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_834 = getitem_95 = primals_172 = None
    getitem_251: "f32[8, 256, 16, 16]" = convolution_backward_29[0]
    getitem_252: "f32[256, 256, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_369: "f32[8, 256, 16, 16]" = torch.ops.aten.add.Tensor(add_368, getitem_251);  add_368 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_6: "f32[8, 512, 16, 16]" = torch.ops.aten.cat.default([slice_3, add_369], 1);  slice_3 = add_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_836: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(cat_6, 0.01)
    where_97: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_97, cat_6, mul_836);  gt_97 = cat_6 = mul_836 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_97, [0, 2, 3])
    sub_187: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_630);  convolution_36 = unsqueeze_630 = None
    mul_837: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_97, sub_187)
    sum_63: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_837, [0, 2, 3]);  mul_837 = None
    mul_838: "f32[512]" = torch.ops.aten.mul.Tensor(sum_62, 0.00048828125)
    unsqueeze_631: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_632: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_839: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, 0.00048828125)
    mul_840: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_841: "f32[512]" = torch.ops.aten.mul.Tensor(mul_839, mul_840);  mul_839 = mul_840 = None
    unsqueeze_634: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_635: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_842: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_637: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_638: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_843: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_636);  sub_187 = unsqueeze_636 = None
    sub_189: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_97, mul_843);  where_97 = mul_843 = None
    sub_190: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_633);  sub_189 = unsqueeze_633 = None
    mul_844: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_639);  sub_190 = unsqueeze_639 = None
    mul_845: "f32[512]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_109);  sum_63 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_844, where_35, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_844 = primals_171 = None
    getitem_254: "f32[8, 512, 16, 16]" = convolution_backward_30[0]
    getitem_255: "f32[512, 512, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_161: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(where_35);  where_35 = None
    alias_162: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    gt_98: "b8[8, 512, 16, 16]" = torch.ops.aten.gt.Scalar(alias_162, 0);  alias_162 = None
    mul_846: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_254, 0.01)
    where_98: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(gt_98, getitem_254, mul_846);  gt_98 = getitem_254 = mul_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_98, [0, 2, 3])
    sub_191: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_642);  convolution_35 = unsqueeze_642 = None
    mul_847: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_98, sub_191)
    sum_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 2, 3]);  mul_847 = None
    mul_848: "f32[512]" = torch.ops.aten.mul.Tensor(sum_64, 0.00048828125)
    unsqueeze_643: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_644: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_849: "f32[512]" = torch.ops.aten.mul.Tensor(sum_65, 0.00048828125)
    mul_850: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_851: "f32[512]" = torch.ops.aten.mul.Tensor(mul_849, mul_850);  mul_849 = mul_850 = None
    unsqueeze_646: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_851, 0);  mul_851 = None
    unsqueeze_647: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_852: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_649: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_650: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_853: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_648);  sub_191 = unsqueeze_648 = None
    sub_193: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_98, mul_853);  where_98 = mul_853 = None
    sub_194: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_645);  sub_193 = unsqueeze_645 = None
    mul_854: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_651);  sub_194 = unsqueeze_651 = None
    mul_855: "f32[512]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_106);  sum_65 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_854, where_34, primals_170, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_854 = primals_170 = None
    getitem_257: "f32[8, 256, 32, 32]" = convolution_backward_31[0]
    getitem_258: "f32[512, 256, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_164: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_34);  where_34 = None
    alias_165: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    gt_99: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_165, 0);  alias_165 = None
    mul_856: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_257, 0.01)
    where_99: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_99, getitem_257, mul_856);  gt_99 = getitem_257 = mul_856 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_99, [0, 2, 3])
    sub_195: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_654);  convolution_34 = unsqueeze_654 = None
    mul_857: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_99, sub_195)
    sum_67: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3]);  mul_857 = None
    mul_858: "f32[256]" = torch.ops.aten.mul.Tensor(sum_66, 0.0001220703125)
    unsqueeze_655: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_656: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_859: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, 0.0001220703125)
    mul_860: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_861: "f32[256]" = torch.ops.aten.mul.Tensor(mul_859, mul_860);  mul_859 = mul_860 = None
    unsqueeze_658: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_659: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_862: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_661: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_662: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_863: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_660);  sub_195 = unsqueeze_660 = None
    sub_197: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_99, mul_863);  where_99 = mul_863 = None
    sub_198: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_657);  sub_197 = unsqueeze_657 = None
    mul_864: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_663);  sub_198 = unsqueeze_663 = None
    mul_865: "f32[256]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_103);  sum_67 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_864, cat_2, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_864 = cat_2 = primals_169 = None
    getitem_260: "f32[8, 256, 32, 32]" = convolution_backward_32[0]
    getitem_261: "f32[256, 256, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_5: "f32[8, 128, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_260, 1, 0, 128)
    slice_6: "f32[8, 128, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_260, 1, 128, 256);  getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_866: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(slice_6, 0.01)
    where_100: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_100, slice_6, mul_866);  gt_100 = slice_6 = mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_100, [0, 2, 3])
    sub_199: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_666);  convolution_33 = unsqueeze_666 = None
    mul_867: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_100, sub_199)
    sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
    mul_868: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 0.0001220703125)
    unsqueeze_667: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_668: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_869: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, 0.0001220703125)
    mul_870: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_871: "f32[128]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_670: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_671: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_872: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_673: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_674: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_873: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_672);  sub_199 = unsqueeze_672 = None
    sub_201: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_100, mul_873);  where_100 = mul_873 = None
    sub_202: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_669);  sub_201 = unsqueeze_669 = None
    mul_874: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_675);  sub_202 = unsqueeze_675 = None
    mul_875: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_100);  sum_69 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_874, add_175, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_874 = add_175 = primals_168 = None
    getitem_263: "f32[8, 128, 32, 32]" = convolution_backward_33[0]
    getitem_264: "f32[128, 128, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_876: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_263, 0.01)
    where_101: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_101, getitem_263, mul_876);  gt_101 = mul_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_101, [0, 2, 3])
    sub_203: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_678);  convolution_32 = unsqueeze_678 = None
    mul_877: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_101, sub_203)
    sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
    mul_878: "f32[128]" = torch.ops.aten.mul.Tensor(sum_70, 0.0001220703125)
    unsqueeze_679: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_878, 0);  mul_878 = None
    unsqueeze_680: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_879: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, 0.0001220703125)
    mul_880: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_881: "f32[128]" = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    unsqueeze_682: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_683: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_882: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_685: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_686: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_883: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_684);  sub_203 = unsqueeze_684 = None
    sub_205: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_101, mul_883);  where_101 = mul_883 = None
    sub_206: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_681);  sub_205 = unsqueeze_681 = None
    mul_884: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_687);  sub_206 = unsqueeze_687 = None
    mul_885: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_97);  sum_71 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_884, where_31, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_884 = primals_167 = None
    getitem_266: "f32[8, 128, 32, 32]" = convolution_backward_34[0]
    getitem_267: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_173: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_31);  where_31 = None
    alias_174: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_173);  alias_173 = None
    gt_102: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_174, 0);  alias_174 = None
    mul_886: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_266, 0.01)
    where_102: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_102, getitem_266, mul_886);  gt_102 = getitem_266 = mul_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_102, [0, 2, 3])
    sub_207: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_690);  convolution_31 = unsqueeze_690 = None
    mul_887: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_102, sub_207)
    sum_73: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(sum_72, 0.0001220703125)
    unsqueeze_691: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_692: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_889: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, 0.0001220703125)
    mul_890: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_891: "f32[128]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    unsqueeze_694: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_695: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_892: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_697: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_698: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_893: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_696);  sub_207 = unsqueeze_696 = None
    sub_209: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_102, mul_893);  where_102 = mul_893 = None
    sub_210: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_693);  sub_209 = unsqueeze_693 = None
    mul_894: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_699);  sub_210 = unsqueeze_699 = None
    mul_895: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_94);  sum_73 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_894, add_164, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_894 = add_164 = primals_166 = None
    getitem_269: "f32[8, 128, 32, 32]" = convolution_backward_35[0]
    getitem_270: "f32[128, 128, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_370: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(getitem_263, getitem_269);  getitem_263 = getitem_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_896: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_370, 0.01)
    where_103: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_103, add_370, mul_896);  gt_103 = mul_896 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_103, [0, 2, 3])
    sub_211: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_702);  convolution_30 = unsqueeze_702 = None
    mul_897: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_103, sub_211)
    sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[128]" = torch.ops.aten.mul.Tensor(sum_74, 0.0001220703125)
    unsqueeze_703: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_704: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_899: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, 0.0001220703125)
    mul_900: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_901: "f32[128]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_706: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_707: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_902: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_709: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_710: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_903: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_708);  sub_211 = unsqueeze_708 = None
    sub_213: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_103, mul_903);  where_103 = mul_903 = None
    sub_214: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_705);  sub_213 = unsqueeze_705 = None
    mul_904: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_711);  sub_214 = unsqueeze_711 = None
    mul_905: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_91);  sum_75 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_904, where_29, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = primals_165 = None
    getitem_272: "f32[8, 128, 32, 32]" = convolution_backward_36[0]
    getitem_273: "f32[128, 128, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_179: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_29);  where_29 = None
    alias_180: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_179);  alias_179 = None
    gt_104: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_180, 0);  alias_180 = None
    mul_906: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_272, 0.01)
    where_104: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_104, getitem_272, mul_906);  gt_104 = getitem_272 = mul_906 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_104, [0, 2, 3])
    sub_215: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_714);  convolution_29 = unsqueeze_714 = None
    mul_907: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_104, sub_215)
    sum_77: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_907, [0, 2, 3]);  mul_907 = None
    mul_908: "f32[128]" = torch.ops.aten.mul.Tensor(sum_76, 0.0001220703125)
    unsqueeze_715: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_716: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_909: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, 0.0001220703125)
    mul_910: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_911: "f32[128]" = torch.ops.aten.mul.Tensor(mul_909, mul_910);  mul_909 = mul_910 = None
    unsqueeze_718: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_719: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_912: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_721: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_722: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_913: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_720);  sub_215 = unsqueeze_720 = None
    sub_217: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_104, mul_913);  where_104 = mul_913 = None
    sub_218: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_217, unsqueeze_717);  sub_217 = unsqueeze_717 = None
    mul_914: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_218, unsqueeze_723);  sub_218 = unsqueeze_723 = None
    mul_915: "f32[128]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_88);  sum_77 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_914, add_153, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_914 = add_153 = primals_164 = None
    getitem_275: "f32[8, 128, 32, 32]" = convolution_backward_37[0]
    getitem_276: "f32[128, 128, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_371: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_370, getitem_275);  add_370 = getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_916: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_371, 0.01)
    where_105: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_105, add_371, mul_916);  gt_105 = mul_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_105, [0, 2, 3])
    sub_219: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_726);  convolution_28 = unsqueeze_726 = None
    mul_917: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_105, sub_219)
    sum_79: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_917, [0, 2, 3]);  mul_917 = None
    mul_918: "f32[128]" = torch.ops.aten.mul.Tensor(sum_78, 0.0001220703125)
    unsqueeze_727: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_728: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_919: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, 0.0001220703125)
    mul_920: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_921: "f32[128]" = torch.ops.aten.mul.Tensor(mul_919, mul_920);  mul_919 = mul_920 = None
    unsqueeze_730: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_731: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_922: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_733: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_922, 0);  mul_922 = None
    unsqueeze_734: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_923: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_732);  sub_219 = unsqueeze_732 = None
    sub_221: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_105, mul_923);  where_105 = mul_923 = None
    sub_222: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_221, unsqueeze_729);  sub_221 = unsqueeze_729 = None
    mul_924: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_735);  sub_222 = unsqueeze_735 = None
    mul_925: "f32[128]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_85);  sum_79 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_924, where_27, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_924 = primals_163 = None
    getitem_278: "f32[8, 128, 32, 32]" = convolution_backward_38[0]
    getitem_279: "f32[128, 128, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_185: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_27);  where_27 = None
    alias_186: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_185);  alias_185 = None
    gt_106: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_186, 0);  alias_186 = None
    mul_926: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_278, 0.01)
    where_106: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_106, getitem_278, mul_926);  gt_106 = getitem_278 = mul_926 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_106, [0, 2, 3])
    sub_223: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_738);  convolution_27 = unsqueeze_738 = None
    mul_927: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_106, sub_223)
    sum_81: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_927, [0, 2, 3]);  mul_927 = None
    mul_928: "f32[128]" = torch.ops.aten.mul.Tensor(sum_80, 0.0001220703125)
    unsqueeze_739: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_740: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_929: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, 0.0001220703125)
    mul_930: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_931: "f32[128]" = torch.ops.aten.mul.Tensor(mul_929, mul_930);  mul_929 = mul_930 = None
    unsqueeze_742: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_931, 0);  mul_931 = None
    unsqueeze_743: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_932: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_745: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_746: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_933: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_744);  sub_223 = unsqueeze_744 = None
    sub_225: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_106, mul_933);  where_106 = mul_933 = None
    sub_226: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_225, unsqueeze_741);  sub_225 = unsqueeze_741 = None
    mul_934: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_747);  sub_226 = unsqueeze_747 = None
    mul_935: "f32[128]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_82);  sum_81 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_934, add_142, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_934 = add_142 = primals_162 = None
    getitem_281: "f32[8, 128, 32, 32]" = convolution_backward_39[0]
    getitem_282: "f32[128, 128, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_372: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_371, getitem_281);  add_371 = getitem_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_936: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_372, 0.01)
    where_107: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_107, add_372, mul_936);  gt_107 = mul_936 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_107, [0, 2, 3])
    sub_227: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_750);  convolution_26 = unsqueeze_750 = None
    mul_937: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_107, sub_227)
    sum_83: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_937, [0, 2, 3]);  mul_937 = None
    mul_938: "f32[128]" = torch.ops.aten.mul.Tensor(sum_82, 0.0001220703125)
    unsqueeze_751: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_752: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_939: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, 0.0001220703125)
    mul_940: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_941: "f32[128]" = torch.ops.aten.mul.Tensor(mul_939, mul_940);  mul_939 = mul_940 = None
    unsqueeze_754: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_755: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_942: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_757: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_758: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_943: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_756);  sub_227 = unsqueeze_756 = None
    sub_229: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_107, mul_943);  where_107 = mul_943 = None
    sub_230: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_753);  sub_229 = unsqueeze_753 = None
    mul_944: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_759);  sub_230 = unsqueeze_759 = None
    mul_945: "f32[128]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_79);  sum_83 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_944, where_25, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_944 = primals_161 = None
    getitem_284: "f32[8, 128, 32, 32]" = convolution_backward_40[0]
    getitem_285: "f32[128, 128, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_191: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_25);  where_25 = None
    alias_192: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_191);  alias_191 = None
    gt_108: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_192, 0);  alias_192 = None
    mul_946: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_284, 0.01)
    where_108: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_108, getitem_284, mul_946);  gt_108 = getitem_284 = mul_946 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_108, [0, 2, 3])
    sub_231: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_762);  convolution_25 = unsqueeze_762 = None
    mul_947: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_108, sub_231)
    sum_85: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3]);  mul_947 = None
    mul_948: "f32[128]" = torch.ops.aten.mul.Tensor(sum_84, 0.0001220703125)
    unsqueeze_763: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
    unsqueeze_764: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_949: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, 0.0001220703125)
    mul_950: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_951: "f32[128]" = torch.ops.aten.mul.Tensor(mul_949, mul_950);  mul_949 = mul_950 = None
    unsqueeze_766: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_767: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_952: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_769: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_952, 0);  mul_952 = None
    unsqueeze_770: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_953: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_768);  sub_231 = unsqueeze_768 = None
    sub_233: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_108, mul_953);  where_108 = mul_953 = None
    sub_234: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_765);  sub_233 = unsqueeze_765 = None
    mul_954: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_771);  sub_234 = unsqueeze_771 = None
    mul_955: "f32[128]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_76);  sum_85 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_954, add_131, primals_160, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_954 = add_131 = primals_160 = None
    getitem_287: "f32[8, 128, 32, 32]" = convolution_backward_41[0]
    getitem_288: "f32[128, 128, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_373: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_372, getitem_287);  add_372 = getitem_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_956: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_373, 0.01)
    where_109: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_109, add_373, mul_956);  gt_109 = mul_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_109, [0, 2, 3])
    sub_235: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_774);  convolution_24 = unsqueeze_774 = None
    mul_957: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_109, sub_235)
    sum_87: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_957, [0, 2, 3]);  mul_957 = None
    mul_958: "f32[128]" = torch.ops.aten.mul.Tensor(sum_86, 0.0001220703125)
    unsqueeze_775: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_958, 0);  mul_958 = None
    unsqueeze_776: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_959: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, 0.0001220703125)
    mul_960: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_961: "f32[128]" = torch.ops.aten.mul.Tensor(mul_959, mul_960);  mul_959 = mul_960 = None
    unsqueeze_778: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_961, 0);  mul_961 = None
    unsqueeze_779: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_962: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_781: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_782: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_963: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_780);  sub_235 = unsqueeze_780 = None
    sub_237: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_109, mul_963);  where_109 = mul_963 = None
    sub_238: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_237, unsqueeze_777);  sub_237 = unsqueeze_777 = None
    mul_964: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_783);  sub_238 = unsqueeze_783 = None
    mul_965: "f32[128]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_73);  sum_87 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_964, where_23, primals_159, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_964 = primals_159 = None
    getitem_290: "f32[8, 128, 32, 32]" = convolution_backward_42[0]
    getitem_291: "f32[128, 128, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_197: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_23);  where_23 = None
    alias_198: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_197);  alias_197 = None
    gt_110: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_198, 0);  alias_198 = None
    mul_966: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_290, 0.01)
    where_110: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_110, getitem_290, mul_966);  gt_110 = getitem_290 = mul_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_110, [0, 2, 3])
    sub_239: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_786);  convolution_23 = unsqueeze_786 = None
    mul_967: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_110, sub_239)
    sum_89: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
    mul_968: "f32[128]" = torch.ops.aten.mul.Tensor(sum_88, 0.0001220703125)
    unsqueeze_787: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_968, 0);  mul_968 = None
    unsqueeze_788: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_969: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, 0.0001220703125)
    mul_970: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_971: "f32[128]" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
    unsqueeze_790: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_791: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_972: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_793: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_794: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_973: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_792);  sub_239 = unsqueeze_792 = None
    sub_241: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_110, mul_973);  where_110 = mul_973 = None
    sub_242: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_241, unsqueeze_789);  sub_241 = unsqueeze_789 = None
    mul_974: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_795);  sub_242 = unsqueeze_795 = None
    mul_975: "f32[128]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_70);  sum_89 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_974, add_120, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_974 = add_120 = primals_158 = None
    getitem_293: "f32[8, 128, 32, 32]" = convolution_backward_43[0]
    getitem_294: "f32[128, 128, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_374: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_373, getitem_293);  add_373 = getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_976: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_374, 0.01)
    where_111: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_111, add_374, mul_976);  gt_111 = mul_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_111, [0, 2, 3])
    sub_243: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_798);  convolution_22 = unsqueeze_798 = None
    mul_977: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_111, sub_243)
    sum_91: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3]);  mul_977 = None
    mul_978: "f32[128]" = torch.ops.aten.mul.Tensor(sum_90, 0.0001220703125)
    unsqueeze_799: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_800: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_979: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, 0.0001220703125)
    mul_980: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_981: "f32[128]" = torch.ops.aten.mul.Tensor(mul_979, mul_980);  mul_979 = mul_980 = None
    unsqueeze_802: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_803: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_982: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_805: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_806: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_983: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_804);  sub_243 = unsqueeze_804 = None
    sub_245: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_111, mul_983);  where_111 = mul_983 = None
    sub_246: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_801);  sub_245 = unsqueeze_801 = None
    mul_984: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_807);  sub_246 = unsqueeze_807 = None
    mul_985: "f32[128]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_984, where_21, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_984 = primals_157 = None
    getitem_296: "f32[8, 128, 32, 32]" = convolution_backward_44[0]
    getitem_297: "f32[128, 128, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_203: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_21);  where_21 = None
    alias_204: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_203);  alias_203 = None
    gt_112: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_204, 0);  alias_204 = None
    mul_986: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_296, 0.01)
    where_112: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_112, getitem_296, mul_986);  gt_112 = getitem_296 = mul_986 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_112, [0, 2, 3])
    sub_247: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_810);  convolution_21 = unsqueeze_810 = None
    mul_987: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_112, sub_247)
    sum_93: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_987, [0, 2, 3]);  mul_987 = None
    mul_988: "f32[128]" = torch.ops.aten.mul.Tensor(sum_92, 0.0001220703125)
    unsqueeze_811: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_812: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_989: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, 0.0001220703125)
    mul_990: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_991: "f32[128]" = torch.ops.aten.mul.Tensor(mul_989, mul_990);  mul_989 = mul_990 = None
    unsqueeze_814: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_815: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_992: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_817: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_818: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_993: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_816);  sub_247 = unsqueeze_816 = None
    sub_249: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_112, mul_993);  where_112 = mul_993 = None
    sub_250: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_813);  sub_249 = unsqueeze_813 = None
    mul_994: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_819);  sub_250 = unsqueeze_819 = None
    mul_995: "f32[128]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_64);  sum_93 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_994, add_109, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_994 = add_109 = primals_156 = None
    getitem_299: "f32[8, 128, 32, 32]" = convolution_backward_45[0]
    getitem_300: "f32[128, 128, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_375: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_374, getitem_299);  add_374 = getitem_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_996: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_375, 0.01)
    where_113: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_113, add_375, mul_996);  gt_113 = mul_996 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_113, [0, 2, 3])
    sub_251: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_822);  convolution_20 = unsqueeze_822 = None
    mul_997: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_113, sub_251)
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2, 3]);  mul_997 = None
    mul_998: "f32[128]" = torch.ops.aten.mul.Tensor(sum_94, 0.0001220703125)
    unsqueeze_823: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_824: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_999: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, 0.0001220703125)
    mul_1000: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_1001: "f32[128]" = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    unsqueeze_826: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_827: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_1002: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_829: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_830: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_1003: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_828);  sub_251 = unsqueeze_828 = None
    sub_253: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_113, mul_1003);  where_113 = mul_1003 = None
    sub_254: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_253, unsqueeze_825);  sub_253 = unsqueeze_825 = None
    mul_1004: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_254, unsqueeze_831);  sub_254 = unsqueeze_831 = None
    mul_1005: "f32[128]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_61);  sum_95 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_1004, where_19, primals_155, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1004 = primals_155 = None
    getitem_302: "f32[8, 128, 32, 32]" = convolution_backward_46[0]
    getitem_303: "f32[128, 128, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_209: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_19);  where_19 = None
    alias_210: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_209);  alias_209 = None
    gt_114: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_210, 0);  alias_210 = None
    mul_1006: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_302, 0.01)
    where_114: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_114, getitem_302, mul_1006);  gt_114 = getitem_302 = mul_1006 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_114, [0, 2, 3])
    sub_255: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_834);  convolution_19 = unsqueeze_834 = None
    mul_1007: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_114, sub_255)
    sum_97: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3]);  mul_1007 = None
    mul_1008: "f32[128]" = torch.ops.aten.mul.Tensor(sum_96, 0.0001220703125)
    unsqueeze_835: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_836: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_1009: "f32[128]" = torch.ops.aten.mul.Tensor(sum_97, 0.0001220703125)
    mul_1010: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_1011: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1009, mul_1010);  mul_1009 = mul_1010 = None
    unsqueeze_838: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_839: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_1012: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_841: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
    unsqueeze_842: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_1013: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_840);  sub_255 = unsqueeze_840 = None
    sub_257: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_114, mul_1013);  where_114 = mul_1013 = None
    sub_258: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_837);  sub_257 = unsqueeze_837 = None
    mul_1014: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_843);  sub_258 = unsqueeze_843 = None
    mul_1015: "f32[128]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_58);  sum_97 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_1014, add_98, primals_154, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1014 = add_98 = primals_154 = None
    getitem_305: "f32[8, 128, 32, 32]" = convolution_backward_47[0]
    getitem_306: "f32[128, 128, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_376: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_375, getitem_305);  add_375 = getitem_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1016: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(add_376, 0.01)
    where_115: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_115, add_376, mul_1016);  gt_115 = mul_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_115, [0, 2, 3])
    sub_259: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_846);  convolution_18 = unsqueeze_846 = None
    mul_1017: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_115, sub_259)
    sum_99: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1017, [0, 2, 3]);  mul_1017 = None
    mul_1018: "f32[128]" = torch.ops.aten.mul.Tensor(sum_98, 0.0001220703125)
    unsqueeze_847: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_848: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_1019: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, 0.0001220703125)
    mul_1020: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_1021: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
    unsqueeze_850: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
    unsqueeze_851: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_1022: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_853: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
    unsqueeze_854: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_1023: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_852);  sub_259 = unsqueeze_852 = None
    sub_261: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_115, mul_1023);  where_115 = mul_1023 = None
    sub_262: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_849);  sub_261 = unsqueeze_849 = None
    mul_1024: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_855);  sub_262 = unsqueeze_855 = None
    mul_1025: "f32[128]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_55);  sum_99 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_1024, where_17, primals_153, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1024 = primals_153 = None
    getitem_308: "f32[8, 128, 32, 32]" = convolution_backward_48[0]
    getitem_309: "f32[128, 128, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_215: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(where_17);  where_17 = None
    alias_216: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_215);  alias_215 = None
    gt_116: "b8[8, 128, 32, 32]" = torch.ops.aten.gt.Scalar(alias_216, 0);  alias_216 = None
    mul_1026: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_308, 0.01)
    where_116: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(gt_116, getitem_308, mul_1026);  gt_116 = getitem_308 = mul_1026 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_116, [0, 2, 3])
    sub_263: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_858);  convolution_17 = unsqueeze_858 = None
    mul_1027: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_116, sub_263)
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 2, 3]);  mul_1027 = None
    mul_1028: "f32[128]" = torch.ops.aten.mul.Tensor(sum_100, 0.0001220703125)
    unsqueeze_859: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1028, 0);  mul_1028 = None
    unsqueeze_860: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_1029: "f32[128]" = torch.ops.aten.mul.Tensor(sum_101, 0.0001220703125)
    mul_1030: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_1031: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1029, mul_1030);  mul_1029 = mul_1030 = None
    unsqueeze_862: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_863: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_1032: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_865: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_866: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_1033: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_864);  sub_263 = unsqueeze_864 = None
    sub_265: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_116, mul_1033);  where_116 = mul_1033 = None
    sub_266: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_265, unsqueeze_861);  sub_265 = unsqueeze_861 = None
    mul_1034: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_266, unsqueeze_867);  sub_266 = unsqueeze_867 = None
    mul_1035: "f32[128]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_52);  sum_101 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_1034, getitem_49, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1034 = getitem_49 = primals_152 = None
    getitem_311: "f32[8, 128, 32, 32]" = convolution_backward_49[0]
    getitem_312: "f32[128, 128, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_377: "f32[8, 128, 32, 32]" = torch.ops.aten.add.Tensor(add_376, getitem_311);  add_376 = getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_7: "f32[8, 256, 32, 32]" = torch.ops.aten.cat.default([slice_5, add_377], 1);  slice_5 = add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1036: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(cat_7, 0.01)
    where_117: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_117, cat_7, mul_1036);  gt_117 = cat_7 = mul_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_117, [0, 2, 3])
    sub_267: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_870);  convolution_16 = unsqueeze_870 = None
    mul_1037: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_117, sub_267)
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 2, 3]);  mul_1037 = None
    mul_1038: "f32[256]" = torch.ops.aten.mul.Tensor(sum_102, 0.0001220703125)
    unsqueeze_871: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1038, 0);  mul_1038 = None
    unsqueeze_872: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_1039: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, 0.0001220703125)
    mul_1040: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_1041: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1039, mul_1040);  mul_1039 = mul_1040 = None
    unsqueeze_874: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1041, 0);  mul_1041 = None
    unsqueeze_875: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_1042: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_877: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
    unsqueeze_878: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_1043: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_876);  sub_267 = unsqueeze_876 = None
    sub_269: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_117, mul_1043);  where_117 = mul_1043 = None
    sub_270: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_269, unsqueeze_873);  sub_269 = unsqueeze_873 = None
    mul_1044: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_879);  sub_270 = unsqueeze_879 = None
    mul_1045: "f32[256]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_49);  sum_103 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_1044, where_15, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1044 = primals_151 = None
    getitem_314: "f32[8, 256, 32, 32]" = convolution_backward_50[0]
    getitem_315: "f32[256, 256, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_221: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(where_15);  where_15 = None
    alias_222: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_221);  alias_221 = None
    gt_118: "b8[8, 256, 32, 32]" = torch.ops.aten.gt.Scalar(alias_222, 0);  alias_222 = None
    mul_1046: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_314, 0.01)
    where_118: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(gt_118, getitem_314, mul_1046);  gt_118 = getitem_314 = mul_1046 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_118, [0, 2, 3])
    sub_271: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_882);  convolution_15 = unsqueeze_882 = None
    mul_1047: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_118, sub_271)
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_1047, [0, 2, 3]);  mul_1047 = None
    mul_1048: "f32[256]" = torch.ops.aten.mul.Tensor(sum_104, 0.0001220703125)
    unsqueeze_883: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_884: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_1049: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, 0.0001220703125)
    mul_1050: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_1051: "f32[256]" = torch.ops.aten.mul.Tensor(mul_1049, mul_1050);  mul_1049 = mul_1050 = None
    unsqueeze_886: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1051, 0);  mul_1051 = None
    unsqueeze_887: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_1052: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_889: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
    unsqueeze_890: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_1053: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_888);  sub_271 = unsqueeze_888 = None
    sub_273: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_118, mul_1053);  where_118 = mul_1053 = None
    sub_274: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_273, unsqueeze_885);  sub_273 = unsqueeze_885 = None
    mul_1054: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_891);  sub_274 = unsqueeze_891 = None
    mul_1055: "f32[256]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_46);  sum_105 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_1054, where_14, primals_150, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1054 = primals_150 = None
    getitem_317: "f32[8, 128, 64, 64]" = convolution_backward_51[0]
    getitem_318: "f32[256, 128, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_224: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_14);  where_14 = None
    alias_225: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_224);  alias_224 = None
    gt_119: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_225, 0);  alias_225 = None
    mul_1056: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_317, 0.01)
    where_119: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_119, getitem_317, mul_1056);  gt_119 = getitem_317 = mul_1056 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_119, [0, 2, 3])
    sub_275: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_894);  convolution_14 = unsqueeze_894 = None
    mul_1057: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_119, sub_275)
    sum_107: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1057, [0, 2, 3]);  mul_1057 = None
    mul_1058: "f32[128]" = torch.ops.aten.mul.Tensor(sum_106, 3.0517578125e-05)
    unsqueeze_895: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1058, 0);  mul_1058 = None
    unsqueeze_896: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_1059: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, 3.0517578125e-05)
    mul_1060: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_1061: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1059, mul_1060);  mul_1059 = mul_1060 = None
    unsqueeze_898: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_899: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_1062: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_901: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1062, 0);  mul_1062 = None
    unsqueeze_902: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_1063: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_900);  sub_275 = unsqueeze_900 = None
    sub_277: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_119, mul_1063);  where_119 = mul_1063 = None
    sub_278: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_277, unsqueeze_897);  sub_277 = unsqueeze_897 = None
    mul_1064: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_903);  sub_278 = unsqueeze_903 = None
    mul_1065: "f32[128]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_43);  sum_107 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_1064, cat_1, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1064 = cat_1 = primals_149 = None
    getitem_320: "f32[8, 128, 64, 64]" = convolution_backward_52[0]
    getitem_321: "f32[128, 128, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_7: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(getitem_320, 1, 0, 64)
    slice_8: "f32[8, 64, 64, 64]" = torch.ops.aten.slice.Tensor(getitem_320, 1, 64, 128);  getitem_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1066: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(slice_8, 0.01)
    where_120: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_120, slice_8, mul_1066);  gt_120 = slice_8 = mul_1066 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_120, [0, 2, 3])
    sub_279: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_906);  convolution_13 = unsqueeze_906 = None
    mul_1067: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_120, sub_279)
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1067, [0, 2, 3]);  mul_1067 = None
    mul_1068: "f32[64]" = torch.ops.aten.mul.Tensor(sum_108, 3.0517578125e-05)
    unsqueeze_907: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1068, 0);  mul_1068 = None
    unsqueeze_908: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_1069: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, 3.0517578125e-05)
    mul_1070: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_1071: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1069, mul_1070);  mul_1069 = mul_1070 = None
    unsqueeze_910: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1071, 0);  mul_1071 = None
    unsqueeze_911: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_1072: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_913: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1072, 0);  mul_1072 = None
    unsqueeze_914: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_1073: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_912);  sub_279 = unsqueeze_912 = None
    sub_281: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_120, mul_1073);  where_120 = mul_1073 = None
    sub_282: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_909);  sub_281 = unsqueeze_909 = None
    mul_1074: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_915);  sub_282 = unsqueeze_915 = None
    mul_1075: "f32[64]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_1074, add_67, primals_148, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1074 = add_67 = primals_148 = None
    getitem_323: "f32[8, 64, 64, 64]" = convolution_backward_53[0]
    getitem_324: "f32[64, 64, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1076: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_323, 0.01)
    where_121: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_121, getitem_323, mul_1076);  gt_121 = mul_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_121, [0, 2, 3])
    sub_283: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_918);  convolution_12 = unsqueeze_918 = None
    mul_1077: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_121, sub_283)
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1077, [0, 2, 3]);  mul_1077 = None
    mul_1078: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, 3.0517578125e-05)
    unsqueeze_919: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_920: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_1079: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, 3.0517578125e-05)
    mul_1080: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_1081: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1079, mul_1080);  mul_1079 = mul_1080 = None
    unsqueeze_922: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1081, 0);  mul_1081 = None
    unsqueeze_923: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_1082: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_925: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1082, 0);  mul_1082 = None
    unsqueeze_926: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_1083: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_924);  sub_283 = unsqueeze_924 = None
    sub_285: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_121, mul_1083);  where_121 = mul_1083 = None
    sub_286: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_921);  sub_285 = unsqueeze_921 = None
    mul_1084: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_927);  sub_286 = unsqueeze_927 = None
    mul_1085: "f32[64]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_1084, where_11, primals_147, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1084 = primals_147 = None
    getitem_326: "f32[8, 64, 64, 64]" = convolution_backward_54[0]
    getitem_327: "f32[64, 64, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_233: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_11);  where_11 = None
    alias_234: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_233);  alias_233 = None
    gt_122: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_234, 0);  alias_234 = None
    mul_1086: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_326, 0.01)
    where_122: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_122, getitem_326, mul_1086);  gt_122 = getitem_326 = mul_1086 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_122, [0, 2, 3])
    sub_287: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_930);  convolution_11 = unsqueeze_930 = None
    mul_1087: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_122, sub_287)
    sum_113: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1087, [0, 2, 3]);  mul_1087 = None
    mul_1088: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
    unsqueeze_931: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1088, 0);  mul_1088 = None
    unsqueeze_932: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_1089: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
    mul_1090: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1091: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1089, mul_1090);  mul_1089 = mul_1090 = None
    unsqueeze_934: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_935: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_1092: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_937: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1092, 0);  mul_1092 = None
    unsqueeze_938: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_1093: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_936);  sub_287 = unsqueeze_936 = None
    sub_289: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_122, mul_1093);  where_122 = mul_1093 = None
    sub_290: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_289, unsqueeze_933);  sub_289 = unsqueeze_933 = None
    mul_1094: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_939);  sub_290 = unsqueeze_939 = None
    mul_1095: "f32[64]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_1094, add_56, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1094 = add_56 = primals_146 = None
    getitem_329: "f32[8, 64, 64, 64]" = convolution_backward_55[0]
    getitem_330: "f32[64, 64, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_378: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_323, getitem_329);  getitem_323 = getitem_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1096: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_378, 0.01)
    where_123: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_123, add_378, mul_1096);  gt_123 = mul_1096 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_123, [0, 2, 3])
    sub_291: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_942);  convolution_10 = unsqueeze_942 = None
    mul_1097: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_123, sub_291)
    sum_115: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1097, [0, 2, 3]);  mul_1097 = None
    mul_1098: "f32[64]" = torch.ops.aten.mul.Tensor(sum_114, 3.0517578125e-05)
    unsqueeze_943: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1098, 0);  mul_1098 = None
    unsqueeze_944: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_1099: "f32[64]" = torch.ops.aten.mul.Tensor(sum_115, 3.0517578125e-05)
    mul_1100: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1101: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1099, mul_1100);  mul_1099 = mul_1100 = None
    unsqueeze_946: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1101, 0);  mul_1101 = None
    unsqueeze_947: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_1102: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_949: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1102, 0);  mul_1102 = None
    unsqueeze_950: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_1103: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_948);  sub_291 = unsqueeze_948 = None
    sub_293: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_123, mul_1103);  where_123 = mul_1103 = None
    sub_294: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_293, unsqueeze_945);  sub_293 = unsqueeze_945 = None
    mul_1104: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_951);  sub_294 = unsqueeze_951 = None
    mul_1105: "f32[64]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_31);  sum_115 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_1104, where_9, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1104 = primals_145 = None
    getitem_332: "f32[8, 64, 64, 64]" = convolution_backward_56[0]
    getitem_333: "f32[64, 64, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_239: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(where_9);  where_9 = None
    alias_240: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_239);  alias_239 = None
    gt_124: "b8[8, 64, 64, 64]" = torch.ops.aten.gt.Scalar(alias_240, 0);  alias_240 = None
    mul_1106: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_332, 0.01)
    where_124: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(gt_124, getitem_332, mul_1106);  gt_124 = getitem_332 = mul_1106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_124, [0, 2, 3])
    sub_295: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_954);  convolution_9 = unsqueeze_954 = None
    mul_1107: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_124, sub_295)
    sum_117: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1107, [0, 2, 3]);  mul_1107 = None
    mul_1108: "f32[64]" = torch.ops.aten.mul.Tensor(sum_116, 3.0517578125e-05)
    unsqueeze_955: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_956: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_1109: "f32[64]" = torch.ops.aten.mul.Tensor(sum_117, 3.0517578125e-05)
    mul_1110: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1111: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1109, mul_1110);  mul_1109 = mul_1110 = None
    unsqueeze_958: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1111, 0);  mul_1111 = None
    unsqueeze_959: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_1112: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_961: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1112, 0);  mul_1112 = None
    unsqueeze_962: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_1113: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_960);  sub_295 = unsqueeze_960 = None
    sub_297: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_124, mul_1113);  where_124 = mul_1113 = None
    sub_298: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_297, unsqueeze_957);  sub_297 = unsqueeze_957 = None
    mul_1114: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_298, unsqueeze_963);  sub_298 = unsqueeze_963 = None
    mul_1115: "f32[64]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_28);  sum_117 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_1114, getitem_27, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1114 = getitem_27 = primals_144 = None
    getitem_335: "f32[8, 64, 64, 64]" = convolution_backward_57[0]
    getitem_336: "f32[64, 64, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_379: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_378, getitem_335);  add_378 = getitem_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_8: "f32[8, 128, 64, 64]" = torch.ops.aten.cat.default([slice_7, add_379], 1);  slice_7 = add_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1116: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(cat_8, 0.01)
    where_125: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_125, cat_8, mul_1116);  gt_125 = cat_8 = mul_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_125, [0, 2, 3])
    sub_299: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_966);  convolution_8 = unsqueeze_966 = None
    mul_1117: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_125, sub_299)
    sum_119: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 2, 3]);  mul_1117 = None
    mul_1118: "f32[128]" = torch.ops.aten.mul.Tensor(sum_118, 3.0517578125e-05)
    unsqueeze_967: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1118, 0);  mul_1118 = None
    unsqueeze_968: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_1119: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, 3.0517578125e-05)
    mul_1120: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1121: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1119, mul_1120);  mul_1119 = mul_1120 = None
    unsqueeze_970: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_971: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_1122: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_973: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1122, 0);  mul_1122 = None
    unsqueeze_974: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_1123: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_972);  sub_299 = unsqueeze_972 = None
    sub_301: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_125, mul_1123);  where_125 = mul_1123 = None
    sub_302: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_301, unsqueeze_969);  sub_301 = unsqueeze_969 = None
    mul_1124: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_302, unsqueeze_975);  sub_302 = unsqueeze_975 = None
    mul_1125: "f32[128]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_25);  sum_119 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_1124, where_7, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1124 = primals_143 = None
    getitem_338: "f32[8, 128, 64, 64]" = convolution_backward_58[0]
    getitem_339: "f32[128, 128, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_245: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(where_7);  where_7 = None
    alias_246: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_245);  alias_245 = None
    gt_126: "b8[8, 128, 64, 64]" = torch.ops.aten.gt.Scalar(alias_246, 0);  alias_246 = None
    mul_1126: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_338, 0.01)
    where_126: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(gt_126, getitem_338, mul_1126);  gt_126 = getitem_338 = mul_1126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_126, [0, 2, 3])
    sub_303: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_978);  convolution_7 = unsqueeze_978 = None
    mul_1127: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_126, sub_303)
    sum_121: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1127, [0, 2, 3]);  mul_1127 = None
    mul_1128: "f32[128]" = torch.ops.aten.mul.Tensor(sum_120, 3.0517578125e-05)
    unsqueeze_979: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1128, 0);  mul_1128 = None
    unsqueeze_980: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_1129: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, 3.0517578125e-05)
    mul_1130: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1131: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1129, mul_1130);  mul_1129 = mul_1130 = None
    unsqueeze_982: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1131, 0);  mul_1131 = None
    unsqueeze_983: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    mul_1132: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_985: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1132, 0);  mul_1132 = None
    unsqueeze_986: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_1133: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_984);  sub_303 = unsqueeze_984 = None
    sub_305: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_126, mul_1133);  where_126 = mul_1133 = None
    sub_306: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_305, unsqueeze_981);  sub_305 = unsqueeze_981 = None
    mul_1134: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_987);  sub_306 = unsqueeze_987 = None
    mul_1135: "f32[128]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_22);  sum_121 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_1134, where_6, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1134 = primals_142 = None
    getitem_341: "f32[8, 64, 128, 128]" = convolution_backward_59[0]
    getitem_342: "f32[128, 64, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_248: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_6);  where_6 = None
    alias_249: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_248);  alias_248 = None
    gt_127: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_249, 0);  alias_249 = None
    mul_1136: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_341, 0.01)
    where_127: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_127, getitem_341, mul_1136);  gt_127 = getitem_341 = mul_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_127, [0, 2, 3])
    sub_307: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_990);  convolution_6 = unsqueeze_990 = None
    mul_1137: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_127, sub_307)
    sum_123: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1137, [0, 2, 3]);  mul_1137 = None
    mul_1138: "f32[64]" = torch.ops.aten.mul.Tensor(sum_122, 7.62939453125e-06)
    unsqueeze_991: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1138, 0);  mul_1138 = None
    unsqueeze_992: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_1139: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, 7.62939453125e-06)
    mul_1140: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1141: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1139, mul_1140);  mul_1139 = mul_1140 = None
    unsqueeze_994: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1141, 0);  mul_1141 = None
    unsqueeze_995: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    mul_1142: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_997: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1142, 0);  mul_1142 = None
    unsqueeze_998: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_1143: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_996);  sub_307 = unsqueeze_996 = None
    sub_309: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_127, mul_1143);  where_127 = mul_1143 = None
    sub_310: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_309, unsqueeze_993);  sub_309 = unsqueeze_993 = None
    mul_1144: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_999);  sub_310 = unsqueeze_999 = None
    mul_1145: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_19);  sum_123 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1144, cat, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1144 = cat = primals_141 = None
    getitem_344: "f32[8, 128, 128, 128]" = convolution_backward_60[0]
    getitem_345: "f32[64, 128, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:339, code: out = self.conv_transition(torch.cat([xs, xb], dim=1))
    slice_9: "f32[8, 64, 128, 128]" = torch.ops.aten.slice.Tensor(getitem_344, 1, 0, 64)
    slice_10: "f32[8, 64, 128, 128]" = torch.ops.aten.slice.Tensor(getitem_344, 1, 64, 128);  getitem_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1146: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(slice_10, 0.01)
    where_128: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_128, slice_10, mul_1146);  gt_128 = slice_10 = mul_1146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_128, [0, 2, 3])
    sub_311: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_1002);  convolution_5 = unsqueeze_1002 = None
    mul_1147: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_128, sub_311)
    sum_125: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1147, [0, 2, 3]);  mul_1147 = None
    mul_1148: "f32[64]" = torch.ops.aten.mul.Tensor(sum_124, 7.62939453125e-06)
    unsqueeze_1003: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1148, 0);  mul_1148 = None
    unsqueeze_1004: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1149: "f32[64]" = torch.ops.aten.mul.Tensor(sum_125, 7.62939453125e-06)
    mul_1150: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1151: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1149, mul_1150);  mul_1149 = mul_1150 = None
    unsqueeze_1006: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1151, 0);  mul_1151 = None
    unsqueeze_1007: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    mul_1152: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_1009: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1152, 0);  mul_1152 = None
    unsqueeze_1010: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_1153: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_1008);  sub_311 = unsqueeze_1008 = None
    sub_313: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_128, mul_1153);  where_128 = mul_1153 = None
    sub_314: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_313, unsqueeze_1005);  sub_313 = unsqueeze_1005 = None
    mul_1154: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_1011);  sub_314 = unsqueeze_1011 = None
    mul_1155: "f32[64]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_16);  sum_125 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1154, add_25, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1154 = add_25 = primals_140 = None
    getitem_347: "f32[8, 64, 128, 128]" = convolution_backward_61[0]
    getitem_348: "f32[64, 64, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1156: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_347, 0.01)
    where_129: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_129, getitem_347, mul_1156);  gt_129 = mul_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_129, [0, 2, 3])
    sub_315: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_1014);  convolution_4 = unsqueeze_1014 = None
    mul_1157: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_129, sub_315)
    sum_127: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1157, [0, 2, 3]);  mul_1157 = None
    mul_1158: "f32[64]" = torch.ops.aten.mul.Tensor(sum_126, 7.62939453125e-06)
    unsqueeze_1015: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1158, 0);  mul_1158 = None
    unsqueeze_1016: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1159: "f32[64]" = torch.ops.aten.mul.Tensor(sum_127, 7.62939453125e-06)
    mul_1160: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1161: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1159, mul_1160);  mul_1159 = mul_1160 = None
    unsqueeze_1018: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1161, 0);  mul_1161 = None
    unsqueeze_1019: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    mul_1162: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_1021: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1162, 0);  mul_1162 = None
    unsqueeze_1022: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_1163: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1020);  sub_315 = unsqueeze_1020 = None
    sub_317: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_129, mul_1163);  where_129 = mul_1163 = None
    sub_318: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_317, unsqueeze_1017);  sub_317 = unsqueeze_1017 = None
    mul_1164: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_1023);  sub_318 = unsqueeze_1023 = None
    mul_1165: "f32[64]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_13);  sum_127 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1164, where_3, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1164 = primals_139 = None
    getitem_350: "f32[8, 32, 128, 128]" = convolution_backward_62[0]
    getitem_351: "f32[64, 32, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_257: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(where_3);  where_3 = None
    alias_258: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(alias_257);  alias_257 = None
    gt_130: "b8[8, 32, 128, 128]" = torch.ops.aten.gt.Scalar(alias_258, 0);  alias_258 = None
    mul_1166: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_350, 0.01)
    where_130: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(gt_130, getitem_350, mul_1166);  gt_130 = getitem_350 = mul_1166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_130, [0, 2, 3])
    sub_319: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_1026);  convolution_3 = unsqueeze_1026 = None
    mul_1167: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(where_130, sub_319)
    sum_129: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1167, [0, 2, 3]);  mul_1167 = None
    mul_1168: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, 7.62939453125e-06)
    unsqueeze_1027: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1168, 0);  mul_1168 = None
    unsqueeze_1028: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1027, 2);  unsqueeze_1027 = None
    unsqueeze_1029: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, 3);  unsqueeze_1028 = None
    mul_1169: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, 7.62939453125e-06)
    mul_1170: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1171: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1169, mul_1170);  mul_1169 = mul_1170 = None
    unsqueeze_1030: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1171, 0);  mul_1171 = None
    unsqueeze_1031: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, 2);  unsqueeze_1030 = None
    unsqueeze_1032: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 3);  unsqueeze_1031 = None
    mul_1172: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1033: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1172, 0);  mul_1172 = None
    unsqueeze_1034: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1033, 2);  unsqueeze_1033 = None
    unsqueeze_1035: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 3);  unsqueeze_1034 = None
    mul_1173: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1032);  sub_319 = unsqueeze_1032 = None
    sub_321: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(where_130, mul_1173);  where_130 = mul_1173 = None
    sub_322: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_321, unsqueeze_1029);  sub_321 = unsqueeze_1029 = None
    mul_1174: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_322, unsqueeze_1035);  sub_322 = unsqueeze_1035 = None
    mul_1175: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_10);  sum_129 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1174, getitem_9, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1174 = getitem_9 = primals_138 = None
    getitem_353: "f32[8, 64, 128, 128]" = convolution_backward_63[0]
    getitem_354: "f32[32, 64, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_380: "f32[8, 64, 128, 128]" = torch.ops.aten.add.Tensor(getitem_347, getitem_353);  getitem_347 = getitem_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cspnet.py:336, code: xs, xb = x.split(self.expand_chs // 2, dim=1)
    cat_9: "f32[8, 128, 128, 128]" = torch.ops.aten.cat.default([slice_9, add_380], 1);  slice_9 = add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1176: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(cat_9, 0.01)
    where_131: "f32[8, 128, 128, 128]" = torch.ops.aten.where.self(gt_131, cat_9, mul_1176);  gt_131 = cat_9 = mul_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_131, [0, 2, 3])
    sub_323: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1038);  convolution_2 = unsqueeze_1038 = None
    mul_1177: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(where_131, sub_323)
    sum_131: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_1177, [0, 2, 3]);  mul_1177 = None
    mul_1178: "f32[128]" = torch.ops.aten.mul.Tensor(sum_130, 7.62939453125e-06)
    unsqueeze_1039: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1178, 0);  mul_1178 = None
    unsqueeze_1040: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1039, 2);  unsqueeze_1039 = None
    unsqueeze_1041: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, 3);  unsqueeze_1040 = None
    mul_1179: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, 7.62939453125e-06)
    mul_1180: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1181: "f32[128]" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
    unsqueeze_1042: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1181, 0);  mul_1181 = None
    unsqueeze_1043: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, 2);  unsqueeze_1042 = None
    unsqueeze_1044: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1043, 3);  unsqueeze_1043 = None
    mul_1182: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1045: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_1182, 0);  mul_1182 = None
    unsqueeze_1046: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1045, 2);  unsqueeze_1045 = None
    unsqueeze_1047: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, 3);  unsqueeze_1046 = None
    mul_1183: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_1044);  sub_323 = unsqueeze_1044 = None
    sub_325: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(where_131, mul_1183);  where_131 = mul_1183 = None
    sub_326: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_1041);  sub_325 = unsqueeze_1041 = None
    mul_1184: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_1047);  sub_326 = unsqueeze_1047 = None
    mul_1185: "f32[128]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_7);  sum_131 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1184, where_1, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1184 = primals_137 = None
    getitem_356: "f32[8, 64, 128, 128]" = convolution_backward_64[0]
    getitem_357: "f32[128, 64, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_263: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(where_1);  where_1 = None
    alias_264: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_263);  alias_263 = None
    gt_132: "b8[8, 64, 128, 128]" = torch.ops.aten.gt.Scalar(alias_264, 0);  alias_264 = None
    mul_1186: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_356, 0.01)
    where_132: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(gt_132, getitem_356, mul_1186);  gt_132 = getitem_356 = mul_1186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_132, [0, 2, 3])
    sub_327: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1050);  convolution_1 = unsqueeze_1050 = None
    mul_1187: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_132, sub_327)
    sum_133: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_1187, [0, 2, 3]);  mul_1187 = None
    mul_1188: "f32[64]" = torch.ops.aten.mul.Tensor(sum_132, 7.62939453125e-06)
    unsqueeze_1051: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1188, 0);  mul_1188 = None
    unsqueeze_1052: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1051, 2);  unsqueeze_1051 = None
    unsqueeze_1053: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, 3);  unsqueeze_1052 = None
    mul_1189: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, 7.62939453125e-06)
    mul_1190: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1191: "f32[64]" = torch.ops.aten.mul.Tensor(mul_1189, mul_1190);  mul_1189 = mul_1190 = None
    unsqueeze_1054: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1191, 0);  mul_1191 = None
    unsqueeze_1055: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, 2);  unsqueeze_1054 = None
    unsqueeze_1056: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1055, 3);  unsqueeze_1055 = None
    mul_1192: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1057: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_1192, 0);  mul_1192 = None
    unsqueeze_1058: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1057, 2);  unsqueeze_1057 = None
    unsqueeze_1059: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, 3);  unsqueeze_1058 = None
    mul_1193: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_1056);  sub_327 = unsqueeze_1056 = None
    sub_329: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_132, mul_1193);  where_132 = mul_1193 = None
    sub_330: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_1053);  sub_329 = unsqueeze_1053 = None
    mul_1194: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_1059);  sub_330 = unsqueeze_1059 = None
    mul_1195: "f32[64]" = torch.ops.aten.mul.Tensor(sum_133, squeeze_4);  sum_133 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:126, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1194, where, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1194 = primals_136 = None
    getitem_359: "f32[8, 32, 256, 256]" = convolution_backward_65[0]
    getitem_360: "f32[64, 32, 3, 3]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_266: "f32[8, 32, 256, 256]" = torch.ops.aten.alias.default(where);  where = None
    alias_267: "f32[8, 32, 256, 256]" = torch.ops.aten.alias.default(alias_266);  alias_266 = None
    gt_133: "b8[8, 32, 256, 256]" = torch.ops.aten.gt.Scalar(alias_267, 0);  alias_267 = None
    mul_1196: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(getitem_359, 0.01)
    where_133: "f32[8, 32, 256, 256]" = torch.ops.aten.where.self(gt_133, getitem_359, mul_1196);  gt_133 = getitem_359 = mul_1196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_134: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_133, [0, 2, 3])
    sub_331: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1062);  convolution = unsqueeze_1062 = None
    mul_1197: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(where_133, sub_331)
    sum_135: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1197, [0, 2, 3]);  mul_1197 = None
    mul_1198: "f32[32]" = torch.ops.aten.mul.Tensor(sum_134, 1.9073486328125e-06)
    unsqueeze_1063: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1198, 0);  mul_1198 = None
    unsqueeze_1064: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1063, 2);  unsqueeze_1063 = None
    unsqueeze_1065: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, 3);  unsqueeze_1064 = None
    mul_1199: "f32[32]" = torch.ops.aten.mul.Tensor(sum_135, 1.9073486328125e-06)
    mul_1200: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1201: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1199, mul_1200);  mul_1199 = mul_1200 = None
    unsqueeze_1066: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1201, 0);  mul_1201 = None
    unsqueeze_1067: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, 2);  unsqueeze_1066 = None
    unsqueeze_1068: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1067, 3);  unsqueeze_1067 = None
    mul_1202: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1069: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1202, 0);  mul_1202 = None
    unsqueeze_1070: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1069, 2);  unsqueeze_1069 = None
    unsqueeze_1071: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, 3);  unsqueeze_1070 = None
    mul_1203: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_1068);  sub_331 = unsqueeze_1068 = None
    sub_333: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(where_133, mul_1203);  where_133 = mul_1203 = None
    sub_334: "f32[8, 32, 256, 256]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_1065);  sub_333 = unsqueeze_1065 = None
    mul_1204: "f32[8, 32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_1071);  sub_334 = unsqueeze_1071 = None
    mul_1205: "f32[32]" = torch.ops.aten.mul.Tensor(sum_135, squeeze_1);  sum_135 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1204, primals_405, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1204 = primals_405 = primals_135 = None
    getitem_363: "f32[32, 3, 3, 3]" = convolution_backward_66[1];  convolution_backward_66 = None
    return [mul_1205, sum_134, mul_1195, sum_132, mul_1185, sum_130, mul_1175, sum_128, mul_1165, sum_126, mul_1155, sum_124, mul_1145, sum_122, mul_1135, sum_120, mul_1125, sum_118, mul_1115, sum_116, mul_1105, sum_114, mul_1095, sum_112, mul_1085, sum_110, mul_1075, sum_108, mul_1065, sum_106, mul_1055, sum_104, mul_1045, sum_102, mul_1035, sum_100, mul_1025, sum_98, mul_1015, sum_96, mul_1005, sum_94, mul_995, sum_92, mul_985, sum_90, mul_975, sum_88, mul_965, sum_86, mul_955, sum_84, mul_945, sum_82, mul_935, sum_80, mul_925, sum_78, mul_915, sum_76, mul_905, sum_74, mul_895, sum_72, mul_885, sum_70, mul_875, sum_68, mul_865, sum_66, mul_855, sum_64, mul_845, sum_62, mul_835, sum_60, mul_825, sum_58, mul_815, sum_56, mul_805, sum_54, mul_795, sum_52, mul_785, sum_50, mul_775, sum_48, mul_765, sum_46, mul_755, sum_44, mul_745, sum_42, mul_735, sum_40, mul_725, sum_38, mul_715, sum_36, mul_705, sum_34, mul_695, sum_32, mul_685, sum_30, mul_675, sum_28, mul_665, sum_26, mul_655, sum_24, mul_645, sum_22, mul_635, sum_20, mul_625, sum_18, mul_615, sum_16, mul_605, sum_14, mul_595, sum_12, mul_585, sum_10, mul_575, sum_8, mul_565, sum_6, mul_555, sum_4, mul_545, sum_2, getitem_363, getitem_360, getitem_357, getitem_354, getitem_351, getitem_348, getitem_345, getitem_342, getitem_339, getitem_336, getitem_333, getitem_330, getitem_327, getitem_324, getitem_321, getitem_318, getitem_315, getitem_312, getitem_309, getitem_306, getitem_303, getitem_300, getitem_297, getitem_294, getitem_291, getitem_288, getitem_285, getitem_282, getitem_279, getitem_276, getitem_273, getitem_270, getitem_267, getitem_264, getitem_261, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    