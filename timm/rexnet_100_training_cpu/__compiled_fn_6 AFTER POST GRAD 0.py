from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[16]", primals_7: "f32[96]", primals_9: "f32[96]", primals_11: "f32[27]", primals_13: "f32[162]", primals_15: "f32[162]", primals_17: "f32[38]", primals_19: "f32[228]", primals_21: "f32[228]", primals_23: "f32[50]", primals_25: "f32[300]", primals_27: "f32[300]", primals_29: "f32[61]", primals_31: "f32[366]", primals_33: "f32[366]", primals_35: "f32[72]", primals_37: "f32[432]", primals_39: "f32[432]", primals_41: "f32[84]", primals_43: "f32[504]", primals_45: "f32[504]", primals_47: "f32[95]", primals_49: "f32[570]", primals_51: "f32[570]", primals_53: "f32[106]", primals_55: "f32[636]", primals_57: "f32[636]", primals_59: "f32[117]", primals_61: "f32[702]", primals_63: "f32[702]", primals_65: "f32[128]", primals_67: "f32[768]", primals_69: "f32[768]", primals_71: "f32[140]", primals_73: "f32[840]", primals_75: "f32[840]", primals_77: "f32[151]", primals_79: "f32[906]", primals_81: "f32[906]", primals_83: "f32[162]", primals_85: "f32[972]", primals_87: "f32[972]", primals_89: "f32[174]", primals_91: "f32[1044]", primals_93: "f32[1044]", primals_95: "f32[185]", primals_97: "f32[1280]", primals_99: "f32[32, 3, 3, 3]", primals_100: "f32[32, 1, 3, 3]", primals_101: "f32[16, 32, 1, 1]", primals_102: "f32[96, 16, 1, 1]", primals_103: "f32[96, 1, 3, 3]", primals_104: "f32[27, 96, 1, 1]", primals_105: "f32[162, 27, 1, 1]", primals_106: "f32[162, 1, 3, 3]", primals_107: "f32[38, 162, 1, 1]", primals_108: "f32[228, 38, 1, 1]", primals_109: "f32[228, 1, 3, 3]", primals_110: "f32[19, 228, 1, 1]", primals_112: "f32[19]", primals_114: "f32[228, 19, 1, 1]", primals_116: "f32[50, 228, 1, 1]", primals_117: "f32[300, 50, 1, 1]", primals_118: "f32[300, 1, 3, 3]", primals_119: "f32[25, 300, 1, 1]", primals_121: "f32[25]", primals_123: "f32[300, 25, 1, 1]", primals_125: "f32[61, 300, 1, 1]", primals_126: "f32[366, 61, 1, 1]", primals_127: "f32[366, 1, 3, 3]", primals_128: "f32[30, 366, 1, 1]", primals_130: "f32[30]", primals_132: "f32[366, 30, 1, 1]", primals_134: "f32[72, 366, 1, 1]", primals_135: "f32[432, 72, 1, 1]", primals_136: "f32[432, 1, 3, 3]", primals_137: "f32[36, 432, 1, 1]", primals_139: "f32[36]", primals_141: "f32[432, 36, 1, 1]", primals_143: "f32[84, 432, 1, 1]", primals_144: "f32[504, 84, 1, 1]", primals_145: "f32[504, 1, 3, 3]", primals_146: "f32[42, 504, 1, 1]", primals_148: "f32[42]", primals_150: "f32[504, 42, 1, 1]", primals_152: "f32[95, 504, 1, 1]", primals_153: "f32[570, 95, 1, 1]", primals_154: "f32[570, 1, 3, 3]", primals_155: "f32[47, 570, 1, 1]", primals_157: "f32[47]", primals_159: "f32[570, 47, 1, 1]", primals_161: "f32[106, 570, 1, 1]", primals_162: "f32[636, 106, 1, 1]", primals_163: "f32[636, 1, 3, 3]", primals_164: "f32[53, 636, 1, 1]", primals_166: "f32[53]", primals_168: "f32[636, 53, 1, 1]", primals_170: "f32[117, 636, 1, 1]", primals_171: "f32[702, 117, 1, 1]", primals_172: "f32[702, 1, 3, 3]", primals_173: "f32[58, 702, 1, 1]", primals_175: "f32[58]", primals_177: "f32[702, 58, 1, 1]", primals_179: "f32[128, 702, 1, 1]", primals_180: "f32[768, 128, 1, 1]", primals_181: "f32[768, 1, 3, 3]", primals_182: "f32[64, 768, 1, 1]", primals_184: "f32[64]", primals_186: "f32[768, 64, 1, 1]", primals_188: "f32[140, 768, 1, 1]", primals_189: "f32[840, 140, 1, 1]", primals_190: "f32[840, 1, 3, 3]", primals_191: "f32[70, 840, 1, 1]", primals_193: "f32[70]", primals_195: "f32[840, 70, 1, 1]", primals_197: "f32[151, 840, 1, 1]", primals_198: "f32[906, 151, 1, 1]", primals_199: "f32[906, 1, 3, 3]", primals_200: "f32[75, 906, 1, 1]", primals_202: "f32[75]", primals_204: "f32[906, 75, 1, 1]", primals_206: "f32[162, 906, 1, 1]", primals_207: "f32[972, 162, 1, 1]", primals_208: "f32[972, 1, 3, 3]", primals_209: "f32[81, 972, 1, 1]", primals_211: "f32[81]", primals_213: "f32[972, 81, 1, 1]", primals_215: "f32[174, 972, 1, 1]", primals_216: "f32[1044, 174, 1, 1]", primals_217: "f32[1044, 1, 3, 3]", primals_218: "f32[87, 1044, 1, 1]", primals_220: "f32[87]", primals_222: "f32[1044, 87, 1, 1]", primals_224: "f32[185, 1044, 1, 1]", primals_225: "f32[1280, 185, 1, 1]", primals_414: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", mul_7: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 32, 112, 112]", squeeze_4: "f32[32]", clamp_max: "f32[8, 32, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", add_14: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 96, 112, 112]", squeeze_10: "f32[96]", mul_29: "f32[8, 96, 112, 112]", convolution_4: "f32[8, 96, 56, 56]", squeeze_13: "f32[96]", clamp_max_1: "f32[8, 96, 56, 56]", convolution_5: "f32[8, 27, 56, 56]", squeeze_16: "f32[27]", add_29: "f32[8, 27, 56, 56]", convolution_6: "f32[8, 162, 56, 56]", squeeze_19: "f32[162]", mul_51: "f32[8, 162, 56, 56]", convolution_7: "f32[8, 162, 56, 56]", squeeze_22: "f32[162]", clamp_max_2: "f32[8, 162, 56, 56]", convolution_8: "f32[8, 38, 56, 56]", squeeze_25: "f32[38]", cat: "f32[8, 38, 56, 56]", convolution_9: "f32[8, 228, 56, 56]", squeeze_28: "f32[228]", mul_73: "f32[8, 228, 56, 56]", convolution_10: "f32[8, 228, 28, 28]", squeeze_31: "f32[228]", add_55: "f32[8, 228, 28, 28]", mean: "f32[8, 228, 1, 1]", convolution_11: "f32[8, 19, 1, 1]", relu: "f32[8, 19, 1, 1]", convolution_12: "f32[8, 228, 1, 1]", clamp_max_3: "f32[8, 228, 28, 28]", convolution_13: "f32[8, 50, 28, 28]", squeeze_37: "f32[50]", add_65: "f32[8, 50, 28, 28]", convolution_14: "f32[8, 300, 28, 28]", squeeze_40: "f32[300]", mul_103: "f32[8, 300, 28, 28]", convolution_15: "f32[8, 300, 28, 28]", squeeze_43: "f32[300]", add_75: "f32[8, 300, 28, 28]", mean_1: "f32[8, 300, 1, 1]", convolution_16: "f32[8, 25, 1, 1]", relu_1: "f32[8, 25, 1, 1]", convolution_17: "f32[8, 300, 1, 1]", clamp_max_4: "f32[8, 300, 28, 28]", convolution_18: "f32[8, 61, 28, 28]", squeeze_49: "f32[61]", cat_1: "f32[8, 61, 28, 28]", convolution_19: "f32[8, 366, 28, 28]", squeeze_52: "f32[366]", mul_133: "f32[8, 366, 28, 28]", convolution_20: "f32[8, 366, 14, 14]", squeeze_55: "f32[366]", add_96: "f32[8, 366, 14, 14]", mean_2: "f32[8, 366, 1, 1]", convolution_21: "f32[8, 30, 1, 1]", relu_2: "f32[8, 30, 1, 1]", convolution_22: "f32[8, 366, 1, 1]", clamp_max_5: "f32[8, 366, 14, 14]", convolution_23: "f32[8, 72, 14, 14]", squeeze_61: "f32[72]", add_106: "f32[8, 72, 14, 14]", convolution_24: "f32[8, 432, 14, 14]", squeeze_64: "f32[432]", mul_163: "f32[8, 432, 14, 14]", convolution_25: "f32[8, 432, 14, 14]", squeeze_67: "f32[432]", add_116: "f32[8, 432, 14, 14]", mean_3: "f32[8, 432, 1, 1]", convolution_26: "f32[8, 36, 1, 1]", relu_3: "f32[8, 36, 1, 1]", convolution_27: "f32[8, 432, 1, 1]", clamp_max_6: "f32[8, 432, 14, 14]", convolution_28: "f32[8, 84, 14, 14]", squeeze_73: "f32[84]", cat_2: "f32[8, 84, 14, 14]", convolution_29: "f32[8, 504, 14, 14]", squeeze_76: "f32[504]", mul_193: "f32[8, 504, 14, 14]", convolution_30: "f32[8, 504, 14, 14]", squeeze_79: "f32[504]", add_137: "f32[8, 504, 14, 14]", mean_4: "f32[8, 504, 1, 1]", convolution_31: "f32[8, 42, 1, 1]", relu_4: "f32[8, 42, 1, 1]", convolution_32: "f32[8, 504, 1, 1]", clamp_max_7: "f32[8, 504, 14, 14]", convolution_33: "f32[8, 95, 14, 14]", squeeze_85: "f32[95]", cat_3: "f32[8, 95, 14, 14]", convolution_34: "f32[8, 570, 14, 14]", squeeze_88: "f32[570]", mul_223: "f32[8, 570, 14, 14]", convolution_35: "f32[8, 570, 14, 14]", squeeze_91: "f32[570]", add_158: "f32[8, 570, 14, 14]", mean_5: "f32[8, 570, 1, 1]", convolution_36: "f32[8, 47, 1, 1]", relu_5: "f32[8, 47, 1, 1]", convolution_37: "f32[8, 570, 1, 1]", clamp_max_8: "f32[8, 570, 14, 14]", convolution_38: "f32[8, 106, 14, 14]", squeeze_97: "f32[106]", cat_4: "f32[8, 106, 14, 14]", convolution_39: "f32[8, 636, 14, 14]", squeeze_100: "f32[636]", mul_253: "f32[8, 636, 14, 14]", convolution_40: "f32[8, 636, 14, 14]", squeeze_103: "f32[636]", add_179: "f32[8, 636, 14, 14]", mean_6: "f32[8, 636, 1, 1]", convolution_41: "f32[8, 53, 1, 1]", relu_6: "f32[8, 53, 1, 1]", convolution_42: "f32[8, 636, 1, 1]", clamp_max_9: "f32[8, 636, 14, 14]", convolution_43: "f32[8, 117, 14, 14]", squeeze_109: "f32[117]", cat_5: "f32[8, 117, 14, 14]", convolution_44: "f32[8, 702, 14, 14]", squeeze_112: "f32[702]", mul_283: "f32[8, 702, 14, 14]", convolution_45: "f32[8, 702, 14, 14]", squeeze_115: "f32[702]", add_200: "f32[8, 702, 14, 14]", mean_7: "f32[8, 702, 1, 1]", convolution_46: "f32[8, 58, 1, 1]", relu_7: "f32[8, 58, 1, 1]", convolution_47: "f32[8, 702, 1, 1]", clamp_max_10: "f32[8, 702, 14, 14]", convolution_48: "f32[8, 128, 14, 14]", squeeze_121: "f32[128]", cat_6: "f32[8, 128, 14, 14]", convolution_49: "f32[8, 768, 14, 14]", squeeze_124: "f32[768]", mul_313: "f32[8, 768, 14, 14]", convolution_50: "f32[8, 768, 7, 7]", squeeze_127: "f32[768]", add_221: "f32[8, 768, 7, 7]", mean_8: "f32[8, 768, 1, 1]", convolution_51: "f32[8, 64, 1, 1]", relu_8: "f32[8, 64, 1, 1]", convolution_52: "f32[8, 768, 1, 1]", clamp_max_11: "f32[8, 768, 7, 7]", convolution_53: "f32[8, 140, 7, 7]", squeeze_133: "f32[140]", add_231: "f32[8, 140, 7, 7]", convolution_54: "f32[8, 840, 7, 7]", squeeze_136: "f32[840]", mul_343: "f32[8, 840, 7, 7]", convolution_55: "f32[8, 840, 7, 7]", squeeze_139: "f32[840]", add_241: "f32[8, 840, 7, 7]", mean_9: "f32[8, 840, 1, 1]", convolution_56: "f32[8, 70, 1, 1]", relu_9: "f32[8, 70, 1, 1]", convolution_57: "f32[8, 840, 1, 1]", clamp_max_12: "f32[8, 840, 7, 7]", convolution_58: "f32[8, 151, 7, 7]", squeeze_145: "f32[151]", cat_7: "f32[8, 151, 7, 7]", convolution_59: "f32[8, 906, 7, 7]", squeeze_148: "f32[906]", mul_373: "f32[8, 906, 7, 7]", convolution_60: "f32[8, 906, 7, 7]", squeeze_151: "f32[906]", add_262: "f32[8, 906, 7, 7]", mean_10: "f32[8, 906, 1, 1]", convolution_61: "f32[8, 75, 1, 1]", relu_10: "f32[8, 75, 1, 1]", convolution_62: "f32[8, 906, 1, 1]", clamp_max_13: "f32[8, 906, 7, 7]", convolution_63: "f32[8, 162, 7, 7]", squeeze_157: "f32[162]", cat_8: "f32[8, 162, 7, 7]", convolution_64: "f32[8, 972, 7, 7]", squeeze_160: "f32[972]", mul_403: "f32[8, 972, 7, 7]", convolution_65: "f32[8, 972, 7, 7]", squeeze_163: "f32[972]", add_283: "f32[8, 972, 7, 7]", mean_11: "f32[8, 972, 1, 1]", convolution_66: "f32[8, 81, 1, 1]", relu_11: "f32[8, 81, 1, 1]", convolution_67: "f32[8, 972, 1, 1]", clamp_max_14: "f32[8, 972, 7, 7]", convolution_68: "f32[8, 174, 7, 7]", squeeze_169: "f32[174]", cat_9: "f32[8, 174, 7, 7]", convolution_69: "f32[8, 1044, 7, 7]", squeeze_172: "f32[1044]", mul_433: "f32[8, 1044, 7, 7]", convolution_70: "f32[8, 1044, 7, 7]", squeeze_175: "f32[1044]", add_304: "f32[8, 1044, 7, 7]", mean_12: "f32[8, 1044, 1, 1]", convolution_71: "f32[8, 87, 1, 1]", relu_12: "f32[8, 87, 1, 1]", convolution_72: "f32[8, 1044, 1, 1]", clamp_max_15: "f32[8, 1044, 7, 7]", convolution_73: "f32[8, 185, 7, 7]", squeeze_181: "f32[185]", cat_10: "f32[8, 185, 7, 7]", convolution_74: "f32[8, 1280, 7, 7]", squeeze_184: "f32[1280]", clone_17: "f32[8, 1280]", permute_1: "f32[1000, 1280]", mul_465: "f32[8, 1280, 7, 7]", unsqueeze_250: "f32[1, 1280, 1, 1]", unsqueeze_262: "f32[1, 185, 1, 1]", unsqueeze_286: "f32[1, 1044, 1, 1]", mul_508: "f32[8, 1044, 7, 7]", unsqueeze_298: "f32[1, 1044, 1, 1]", unsqueeze_310: "f32[1, 174, 1, 1]", unsqueeze_334: "f32[1, 972, 1, 1]", mul_551: "f32[8, 972, 7, 7]", unsqueeze_346: "f32[1, 972, 1, 1]", unsqueeze_358: "f32[1, 162, 1, 1]", unsqueeze_382: "f32[1, 906, 1, 1]", mul_594: "f32[8, 906, 7, 7]", unsqueeze_394: "f32[1, 906, 1, 1]", unsqueeze_406: "f32[1, 151, 1, 1]", unsqueeze_430: "f32[1, 840, 1, 1]", mul_637: "f32[8, 840, 7, 7]", unsqueeze_442: "f32[1, 840, 1, 1]", unsqueeze_454: "f32[1, 140, 1, 1]", unsqueeze_478: "f32[1, 768, 1, 1]", mul_680: "f32[8, 768, 14, 14]", unsqueeze_490: "f32[1, 768, 1, 1]", unsqueeze_502: "f32[1, 128, 1, 1]", unsqueeze_526: "f32[1, 702, 1, 1]", mul_723: "f32[8, 702, 14, 14]", unsqueeze_538: "f32[1, 702, 1, 1]", unsqueeze_550: "f32[1, 117, 1, 1]", unsqueeze_574: "f32[1, 636, 1, 1]", mul_766: "f32[8, 636, 14, 14]", unsqueeze_586: "f32[1, 636, 1, 1]", unsqueeze_598: "f32[1, 106, 1, 1]", unsqueeze_622: "f32[1, 570, 1, 1]", mul_809: "f32[8, 570, 14, 14]", unsqueeze_634: "f32[1, 570, 1, 1]", unsqueeze_646: "f32[1, 95, 1, 1]", unsqueeze_670: "f32[1, 504, 1, 1]", mul_852: "f32[8, 504, 14, 14]", unsqueeze_682: "f32[1, 504, 1, 1]", unsqueeze_694: "f32[1, 84, 1, 1]", unsqueeze_718: "f32[1, 432, 1, 1]", mul_895: "f32[8, 432, 14, 14]", unsqueeze_730: "f32[1, 432, 1, 1]", unsqueeze_742: "f32[1, 72, 1, 1]", unsqueeze_766: "f32[1, 366, 1, 1]", mul_938: "f32[8, 366, 28, 28]", unsqueeze_778: "f32[1, 366, 1, 1]", unsqueeze_790: "f32[1, 61, 1, 1]", unsqueeze_814: "f32[1, 300, 1, 1]", mul_981: "f32[8, 300, 28, 28]", unsqueeze_826: "f32[1, 300, 1, 1]", unsqueeze_838: "f32[1, 50, 1, 1]", unsqueeze_862: "f32[1, 228, 1, 1]", mul_1024: "f32[8, 228, 56, 56]", unsqueeze_874: "f32[1, 228, 1, 1]", unsqueeze_886: "f32[1, 38, 1, 1]", bitwise_or_13: "b8[8, 162, 56, 56]", unsqueeze_898: "f32[1, 162, 1, 1]", mul_1054: "f32[8, 162, 56, 56]", unsqueeze_910: "f32[1, 162, 1, 1]", unsqueeze_922: "f32[1, 27, 1, 1]", bitwise_or_14: "b8[8, 96, 56, 56]", unsqueeze_934: "f32[1, 96, 1, 1]", mul_1084: "f32[8, 96, 112, 112]", unsqueeze_946: "f32[1, 96, 1, 1]", unsqueeze_958: "f32[1, 16, 1, 1]", bitwise_or_15: "b8[8, 32, 112, 112]", unsqueeze_970: "f32[1, 32, 1, 1]", mul_1114: "f32[8, 32, 112, 112]", unsqueeze_982: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_11 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
    getitem_22: "f32[1, 19, 1, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 19, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_57: "f32[1, 19, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 19, 1, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    squeeze_33: "f32[19]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
    squeeze_34: "f32[19]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_4: "f32[8, 228, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_12);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_88: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(add_55, sigmoid_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_15 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
    getitem_30: "f32[1, 25, 1, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 25, 1, 1]" = var_mean_15[1];  var_mean_15 = None
    add_77: "f32[1, 25, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 25, 1, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    squeeze_45: "f32[25]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
    squeeze_46: "f32[25]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[8, 300, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_17);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_118: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_75, sigmoid_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_19 = torch.ops.aten.var_mean.correction(convolution_21, [0, 2, 3], correction = 0, keepdim = True)
    getitem_38: "f32[1, 30, 1, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 30, 1, 1]" = var_mean_19[1];  var_mean_19 = None
    add_98: "f32[1, 30, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 30, 1, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    squeeze_57: "f32[30]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
    squeeze_58: "f32[30]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_8: "f32[8, 366, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_148: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(add_96, sigmoid_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_23 = torch.ops.aten.var_mean.correction(convolution_26, [0, 2, 3], correction = 0, keepdim = True)
    getitem_46: "f32[1, 36, 1, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 36, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_118: "f32[1, 36, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 36, 1, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    squeeze_69: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
    squeeze_70: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_10: "f32[8, 432, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_178: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_116, sigmoid_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_27 = torch.ops.aten.var_mean.correction(convolution_31, [0, 2, 3], correction = 0, keepdim = True)
    getitem_54: "f32[1, 42, 1, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 42, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_139: "f32[1, 42, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 42, 1, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    squeeze_81: "f32[42]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
    squeeze_82: "f32[42]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_12: "f32[8, 504, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_208: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_137, sigmoid_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_31 = torch.ops.aten.var_mean.correction(convolution_36, [0, 2, 3], correction = 0, keepdim = True)
    getitem_62: "f32[1, 47, 1, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 47, 1, 1]" = var_mean_31[1];  var_mean_31 = None
    add_160: "f32[1, 47, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 47, 1, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    squeeze_93: "f32[47]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
    squeeze_94: "f32[47]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_14: "f32[8, 570, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_238: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_158, sigmoid_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_35 = torch.ops.aten.var_mean.correction(convolution_41, [0, 2, 3], correction = 0, keepdim = True)
    getitem_70: "f32[1, 53, 1, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 53, 1, 1]" = var_mean_35[1];  var_mean_35 = None
    add_181: "f32[1, 53, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 53, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    squeeze_105: "f32[53]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
    squeeze_106: "f32[53]" = torch.ops.aten.squeeze.dims(rsqrt_35, [0, 2, 3]);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_16: "f32[8, 636, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_268: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_179, sigmoid_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_39 = torch.ops.aten.var_mean.correction(convolution_46, [0, 2, 3], correction = 0, keepdim = True)
    getitem_78: "f32[1, 58, 1, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 58, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_202: "f32[1, 58, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 58, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    squeeze_117: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
    squeeze_118: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_18: "f32[8, 702, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47);  convolution_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_298: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_200, sigmoid_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_43 = torch.ops.aten.var_mean.correction(convolution_51, [0, 2, 3], correction = 0, keepdim = True)
    getitem_86: "f32[1, 64, 1, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 64, 1, 1]" = var_mean_43[1];  var_mean_43 = None
    add_223: "f32[1, 64, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 64, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
    squeeze_129: "f32[64]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
    squeeze_130: "f32[64]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_20: "f32[8, 768, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_328: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_221, sigmoid_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_47 = torch.ops.aten.var_mean.correction(convolution_56, [0, 2, 3], correction = 0, keepdim = True)
    getitem_94: "f32[1, 70, 1, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 70, 1, 1]" = var_mean_47[1];  var_mean_47 = None
    add_243: "f32[1, 70, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 70, 1, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
    squeeze_141: "f32[70]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
    squeeze_142: "f32[70]" = torch.ops.aten.squeeze.dims(rsqrt_47, [0, 2, 3]);  rsqrt_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_22: "f32[8, 840, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57);  convolution_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_358: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_241, sigmoid_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_51 = torch.ops.aten.var_mean.correction(convolution_61, [0, 2, 3], correction = 0, keepdim = True)
    getitem_102: "f32[1, 75, 1, 1]" = var_mean_51[0]
    getitem_103: "f32[1, 75, 1, 1]" = var_mean_51[1];  var_mean_51 = None
    add_264: "f32[1, 75, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_51: "f32[1, 75, 1, 1]" = torch.ops.aten.rsqrt.default(add_264);  add_264 = None
    squeeze_153: "f32[75]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
    squeeze_154: "f32[75]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_24: "f32[8, 906, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62);  convolution_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_388: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_262, sigmoid_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_55 = torch.ops.aten.var_mean.correction(convolution_66, [0, 2, 3], correction = 0, keepdim = True)
    getitem_110: "f32[1, 81, 1, 1]" = var_mean_55[0]
    getitem_111: "f32[1, 81, 1, 1]" = var_mean_55[1];  var_mean_55 = None
    add_285: "f32[1, 81, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_55: "f32[1, 81, 1, 1]" = torch.ops.aten.rsqrt.default(add_285);  add_285 = None
    squeeze_165: "f32[81]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
    squeeze_166: "f32[81]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_26: "f32[8, 972, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67);  convolution_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_418: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_283, sigmoid_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    var_mean_59 = torch.ops.aten.var_mean.correction(convolution_71, [0, 2, 3], correction = 0, keepdim = True)
    getitem_118: "f32[1, 87, 1, 1]" = var_mean_59[0]
    getitem_119: "f32[1, 87, 1, 1]" = var_mean_59[1];  var_mean_59 = None
    add_306: "f32[1, 87, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
    rsqrt_59: "f32[1, 87, 1, 1]" = torch.ops.aten.rsqrt.default(add_306);  add_306 = None
    squeeze_177: "f32[87]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
    squeeze_178: "f32[87]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_28: "f32[8, 1044, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72);  convolution_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_448: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_304, sigmoid_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, clone_17);  permute_2 = clone_17 = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1280, 7, 7]);  view_2 = None
    div: "f32[8, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_466: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div, mul_465);  div = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 2, 3])
    sub_63: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_250);  convolution_74 = unsqueeze_250 = None
    mul_467: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_466, sub_63)
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_251: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_252: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_469: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_470: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_471: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_254: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_255: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_472: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_97);  primals_97 = None
    unsqueeze_257: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_258: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_473: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_256);  sub_63 = unsqueeze_256 = None
    sub_65: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(mul_466, mul_473);  mul_466 = mul_473 = None
    sub_66: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_253);  sub_65 = unsqueeze_253 = None
    mul_474: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_259);  sub_66 = unsqueeze_259 = None
    mul_475: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_184);  sum_3 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_474, cat_10, primals_225, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = cat_10 = primals_225 = None
    getitem_124: "f32[8, 185, 7, 7]" = convolution_backward[0]
    getitem_125: "f32[1280, 185, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_45: "f32[8, 174, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_124, 1, 0, 174)
    slice_46: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_124, 1, 174, 185);  getitem_124 = None
    full_default_1: "f32[8, 185, 7, 7]" = torch.ops.aten.full.default([8, 185, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 185, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_46, 1, 174, 9223372036854775807);  slice_46 = None
    slice_scatter_2: "f32[8, 185, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_1, slice_45, 1, 0, 174);  full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_322: "f32[8, 185, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_2);  slice_scatter = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[185]" = torch.ops.aten.sum.dim_IntList(add_322, [0, 2, 3])
    sub_67: "f32[8, 185, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_262);  convolution_73 = unsqueeze_262 = None
    mul_476: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(add_322, sub_67)
    sum_5: "f32[185]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_477: "f32[185]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_263: "f32[1, 185]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_264: "f32[1, 185, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_478: "f32[185]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_479: "f32[185]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_480: "f32[185]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_266: "f32[1, 185]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_267: "f32[1, 185, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_481: "f32[185]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_95);  primals_95 = None
    unsqueeze_269: "f32[1, 185]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_270: "f32[1, 185, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 185, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_482: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_268);  sub_67 = unsqueeze_268 = None
    sub_69: "f32[8, 185, 7, 7]" = torch.ops.aten.sub.Tensor(add_322, mul_482);  add_322 = mul_482 = None
    sub_70: "f32[8, 185, 7, 7]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_265);  sub_69 = unsqueeze_265 = None
    mul_483: "f32[8, 185, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_271);  sub_70 = unsqueeze_271 = None
    mul_484: "f32[185]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_181);  sum_5 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_483, clamp_max_15, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_483 = clamp_max_15 = primals_224 = None
    getitem_127: "f32[8, 1044, 7, 7]" = convolution_backward_1[0]
    getitem_128: "f32[185, 1044, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le: "b8[8, 1044, 7, 7]" = torch.ops.aten.le.Scalar(mul_448, 0.0)
    ge: "b8[8, 1044, 7, 7]" = torch.ops.aten.ge.Scalar(mul_448, 6.0);  mul_448 = None
    bitwise_or: "b8[8, 1044, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le, ge);  le = ge = None
    full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 1044, 7, 7]" = torch.ops.aten.where.self(bitwise_or, full_default_5, getitem_127);  bitwise_or = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_485: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(where, add_304);  add_304 = None
    mul_486: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(where, sigmoid_28);  where = None
    sum_6: "f32[8, 1044, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2, 3], True);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_71: "f32[8, 1044, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_28)
    mul_487: "f32[8, 1044, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_28, sub_71);  sigmoid_28 = sub_71 = None
    mul_488: "f32[8, 1044, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_487);  sum_6 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_488, relu_12, primals_222, [1044], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_488 = primals_222 = None
    getitem_130: "f32[8, 87, 1, 1]" = convolution_backward_2[0]
    getitem_131: "f32[1044, 87, 1, 1]" = convolution_backward_2[1]
    getitem_132: "f32[1044]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_1: "b8[8, 87, 1, 1]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_1: "f32[8, 87, 1, 1]" = torch.ops.aten.where.self(le_1, full_default_5, getitem_130);  le_1 = getitem_130 = None
    unsqueeze_272: "f32[1, 87]" = torch.ops.aten.unsqueeze.default(squeeze_177, 0);  squeeze_177 = None
    unsqueeze_273: "f32[1, 87, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
    unsqueeze_274: "f32[1, 87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
    sum_7: "f32[87]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_72: "f32[8, 87, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_274);  convolution_71 = unsqueeze_274 = None
    mul_489: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(where_1, sub_72)
    sum_8: "f32[87]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
    mul_490: "f32[87]" = torch.ops.aten.mul.Tensor(sum_7, 0.125)
    unsqueeze_275: "f32[1, 87]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_276: "f32[1, 87, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_491: "f32[87]" = torch.ops.aten.mul.Tensor(sum_8, 0.125)
    mul_492: "f32[87]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_493: "f32[87]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    unsqueeze_278: "f32[1, 87]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
    unsqueeze_279: "f32[1, 87, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_494: "f32[87]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_220);  primals_220 = None
    unsqueeze_281: "f32[1, 87]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_282: "f32[1, 87, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 87, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_495: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_280);  sub_72 = unsqueeze_280 = None
    sub_74: "f32[8, 87, 1, 1]" = torch.ops.aten.sub.Tensor(where_1, mul_495);  where_1 = mul_495 = None
    sub_75: "f32[8, 87, 1, 1]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_277);  sub_74 = unsqueeze_277 = None
    mul_496: "f32[8, 87, 1, 1]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_283);  sub_75 = unsqueeze_283 = None
    mul_497: "f32[87]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_178);  sum_8 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_496, mean_12, primals_218, [87], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_496 = mean_12 = primals_218 = None
    getitem_133: "f32[8, 1044, 1, 1]" = convolution_backward_3[0]
    getitem_134: "f32[87, 1044, 1, 1]" = convolution_backward_3[1]
    getitem_135: "f32[87]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 1044, 7, 7]" = torch.ops.aten.expand.default(getitem_133, [8, 1044, 7, 7]);  getitem_133 = None
    div_1: "f32[8, 1044, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_323: "f32[8, 1044, 7, 7]" = torch.ops.aten.add.Tensor(mul_486, div_1);  mul_486 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_9: "f32[1044]" = torch.ops.aten.sum.dim_IntList(add_323, [0, 2, 3])
    sub_76: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_286);  convolution_70 = unsqueeze_286 = None
    mul_498: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(add_323, sub_76)
    sum_10: "f32[1044]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
    mul_499: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_287: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_288: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_500: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_501: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_502: "f32[1044]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_290: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_291: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_503: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_93);  primals_93 = None
    unsqueeze_293: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_294: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_504: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_292);  sub_76 = unsqueeze_292 = None
    sub_78: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(add_323, mul_504);  add_323 = mul_504 = None
    sub_79: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_289);  sub_78 = unsqueeze_289 = None
    mul_505: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_295);  sub_79 = unsqueeze_295 = None
    mul_506: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_175);  sum_10 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_505, mul_433, primals_217, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1044, [True, True, False]);  mul_505 = mul_433 = primals_217 = None
    getitem_136: "f32[8, 1044, 7, 7]" = convolution_backward_4[0]
    getitem_137: "f32[1044, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_509: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_136, mul_508);  getitem_136 = mul_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_11: "f32[1044]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3])
    sub_81: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_298);  convolution_69 = unsqueeze_298 = None
    mul_510: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(mul_509, sub_81)
    sum_12: "f32[1044]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3]);  mul_510 = None
    mul_511: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_299: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_300: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_512: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_513: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_514: "f32[1044]" = torch.ops.aten.mul.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
    unsqueeze_302: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_303: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_515: "f32[1044]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_91);  primals_91 = None
    unsqueeze_305: "f32[1, 1044]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    unsqueeze_306: "f32[1, 1044, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 1044, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_516: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_304);  sub_81 = unsqueeze_304 = None
    sub_83: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(mul_509, mul_516);  mul_509 = mul_516 = None
    sub_84: "f32[8, 1044, 7, 7]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_301);  sub_83 = unsqueeze_301 = None
    mul_517: "f32[8, 1044, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_307);  sub_84 = unsqueeze_307 = None
    mul_518: "f32[1044]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_172);  sum_12 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_517, cat_9, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = cat_9 = primals_216 = None
    getitem_139: "f32[8, 174, 7, 7]" = convolution_backward_5[0]
    getitem_140: "f32[1044, 174, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_325: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(slice_45, getitem_139);  slice_45 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_47: "f32[8, 162, 7, 7]" = torch.ops.aten.slice.Tensor(add_325, 1, 0, 162)
    slice_48: "f32[8, 12, 7, 7]" = torch.ops.aten.slice.Tensor(add_325, 1, 162, 174);  add_325 = None
    full_default_8: "f32[8, 174, 7, 7]" = torch.ops.aten.full.default([8, 174, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[8, 174, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_48, 1, 162, 9223372036854775807);  slice_48 = None
    slice_scatter_6: "f32[8, 174, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_8, slice_47, 1, 0, 162);  full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_326: "f32[8, 174, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_6);  slice_scatter_4 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_13: "f32[174]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 2, 3])
    sub_85: "f32[8, 174, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_310);  convolution_68 = unsqueeze_310 = None
    mul_519: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(add_326, sub_85)
    sum_14: "f32[174]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
    mul_520: "f32[174]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    unsqueeze_311: "f32[1, 174]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_312: "f32[1, 174, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_521: "f32[174]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    mul_522: "f32[174]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_523: "f32[174]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
    unsqueeze_314: "f32[1, 174]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_315: "f32[1, 174, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_524: "f32[174]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_89);  primals_89 = None
    unsqueeze_317: "f32[1, 174]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_318: "f32[1, 174, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 174, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_525: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_316);  sub_85 = unsqueeze_316 = None
    sub_87: "f32[8, 174, 7, 7]" = torch.ops.aten.sub.Tensor(add_326, mul_525);  add_326 = mul_525 = None
    sub_88: "f32[8, 174, 7, 7]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_313);  sub_87 = unsqueeze_313 = None
    mul_526: "f32[8, 174, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_319);  sub_88 = unsqueeze_319 = None
    mul_527: "f32[174]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_169);  sum_14 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_526, clamp_max_14, primals_215, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_526 = clamp_max_14 = primals_215 = None
    getitem_142: "f32[8, 972, 7, 7]" = convolution_backward_6[0]
    getitem_143: "f32[174, 972, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_2: "b8[8, 972, 7, 7]" = torch.ops.aten.le.Scalar(mul_418, 0.0)
    ge_1: "b8[8, 972, 7, 7]" = torch.ops.aten.ge.Scalar(mul_418, 6.0);  mul_418 = None
    bitwise_or_1: "b8[8, 972, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_2, ge_1);  le_2 = ge_1 = None
    where_2: "f32[8, 972, 7, 7]" = torch.ops.aten.where.self(bitwise_or_1, full_default_5, getitem_142);  bitwise_or_1 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_528: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, add_283);  add_283 = None
    mul_529: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sigmoid_26);  where_2 = None
    sum_15: "f32[8, 972, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2, 3], True);  mul_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_89: "f32[8, 972, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_26)
    mul_530: "f32[8, 972, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_26, sub_89);  sigmoid_26 = sub_89 = None
    mul_531: "f32[8, 972, 1, 1]" = torch.ops.aten.mul.Tensor(sum_15, mul_530);  sum_15 = mul_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_531, relu_11, primals_213, [972], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_531 = primals_213 = None
    getitem_145: "f32[8, 81, 1, 1]" = convolution_backward_7[0]
    getitem_146: "f32[972, 81, 1, 1]" = convolution_backward_7[1]
    getitem_147: "f32[972]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_3: "b8[8, 81, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_3: "f32[8, 81, 1, 1]" = torch.ops.aten.where.self(le_3, full_default_5, getitem_145);  le_3 = getitem_145 = None
    unsqueeze_320: "f32[1, 81]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
    unsqueeze_321: "f32[1, 81, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
    unsqueeze_322: "f32[1, 81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
    sum_16: "f32[81]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_90: "f32[8, 81, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_322);  convolution_66 = unsqueeze_322 = None
    mul_532: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(where_3, sub_90)
    sum_17: "f32[81]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 2, 3]);  mul_532 = None
    mul_533: "f32[81]" = torch.ops.aten.mul.Tensor(sum_16, 0.125)
    unsqueeze_323: "f32[1, 81]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_324: "f32[1, 81, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_534: "f32[81]" = torch.ops.aten.mul.Tensor(sum_17, 0.125)
    mul_535: "f32[81]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_536: "f32[81]" = torch.ops.aten.mul.Tensor(mul_534, mul_535);  mul_534 = mul_535 = None
    unsqueeze_326: "f32[1, 81]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_327: "f32[1, 81, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_537: "f32[81]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_211);  primals_211 = None
    unsqueeze_329: "f32[1, 81]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_330: "f32[1, 81, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 81, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_538: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_328);  sub_90 = unsqueeze_328 = None
    sub_92: "f32[8, 81, 1, 1]" = torch.ops.aten.sub.Tensor(where_3, mul_538);  where_3 = mul_538 = None
    sub_93: "f32[8, 81, 1, 1]" = torch.ops.aten.sub.Tensor(sub_92, unsqueeze_325);  sub_92 = unsqueeze_325 = None
    mul_539: "f32[8, 81, 1, 1]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_331);  sub_93 = unsqueeze_331 = None
    mul_540: "f32[81]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_166);  sum_17 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_539, mean_11, primals_209, [81], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_539 = mean_11 = primals_209 = None
    getitem_148: "f32[8, 972, 1, 1]" = convolution_backward_8[0]
    getitem_149: "f32[81, 972, 1, 1]" = convolution_backward_8[1]
    getitem_150: "f32[81]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 972, 7, 7]" = torch.ops.aten.expand.default(getitem_148, [8, 972, 7, 7]);  getitem_148 = None
    div_2: "f32[8, 972, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_327: "f32[8, 972, 7, 7]" = torch.ops.aten.add.Tensor(mul_529, div_2);  mul_529 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[972]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 2, 3])
    sub_94: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_334);  convolution_65 = unsqueeze_334 = None
    mul_541: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(add_327, sub_94)
    sum_19: "f32[972]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 2, 3]);  mul_541 = None
    mul_542: "f32[972]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_335: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_336: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_543: "f32[972]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_544: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_545: "f32[972]" = torch.ops.aten.mul.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_338: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_339: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_546: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_87);  primals_87 = None
    unsqueeze_341: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_342: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_547: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_340);  sub_94 = unsqueeze_340 = None
    sub_96: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(add_327, mul_547);  add_327 = mul_547 = None
    sub_97: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_337);  sub_96 = unsqueeze_337 = None
    mul_548: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_343);  sub_97 = unsqueeze_343 = None
    mul_549: "f32[972]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_163);  sum_19 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_548, mul_403, primals_208, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 972, [True, True, False]);  mul_548 = mul_403 = primals_208 = None
    getitem_151: "f32[8, 972, 7, 7]" = convolution_backward_9[0]
    getitem_152: "f32[972, 1, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_552: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_151, mul_551);  getitem_151 = mul_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[972]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3])
    sub_99: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_346);  convolution_64 = unsqueeze_346 = None
    mul_553: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(mul_552, sub_99)
    sum_21: "f32[972]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
    mul_554: "f32[972]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_347: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_348: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_555: "f32[972]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_556: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_557: "f32[972]" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
    unsqueeze_350: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_351: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_558: "f32[972]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_85);  primals_85 = None
    unsqueeze_353: "f32[1, 972]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_354: "f32[1, 972, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 972, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_559: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_352);  sub_99 = unsqueeze_352 = None
    sub_101: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(mul_552, mul_559);  mul_552 = mul_559 = None
    sub_102: "f32[8, 972, 7, 7]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_349);  sub_101 = unsqueeze_349 = None
    mul_560: "f32[8, 972, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_355);  sub_102 = unsqueeze_355 = None
    mul_561: "f32[972]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_160);  sum_21 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_560, cat_8, primals_207, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_560 = cat_8 = primals_207 = None
    getitem_154: "f32[8, 162, 7, 7]" = convolution_backward_10[0]
    getitem_155: "f32[972, 162, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_329: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(slice_47, getitem_154);  slice_47 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_49: "f32[8, 151, 7, 7]" = torch.ops.aten.slice.Tensor(add_329, 1, 0, 151)
    slice_50: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(add_329, 1, 151, 162);  add_329 = None
    full_default_15: "f32[8, 162, 7, 7]" = torch.ops.aten.full.default([8, 162, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_8: "f32[8, 162, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_15, slice_50, 1, 151, 9223372036854775807);  slice_50 = None
    slice_scatter_10: "f32[8, 162, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_15, slice_49, 1, 0, 151);  full_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_330: "f32[8, 162, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_10);  slice_scatter_8 = slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[162]" = torch.ops.aten.sum.dim_IntList(add_330, [0, 2, 3])
    sub_103: "f32[8, 162, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_358);  convolution_63 = unsqueeze_358 = None
    mul_562: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(add_330, sub_103)
    sum_23: "f32[162]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_563: "f32[162]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_359: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_360: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_564: "f32[162]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_565: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_566: "f32[162]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_362: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_363: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_567: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_83);  primals_83 = None
    unsqueeze_365: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_366: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_568: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_364);  sub_103 = unsqueeze_364 = None
    sub_105: "f32[8, 162, 7, 7]" = torch.ops.aten.sub.Tensor(add_330, mul_568);  add_330 = mul_568 = None
    sub_106: "f32[8, 162, 7, 7]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_361);  sub_105 = unsqueeze_361 = None
    mul_569: "f32[8, 162, 7, 7]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_367);  sub_106 = unsqueeze_367 = None
    mul_570: "f32[162]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_157);  sum_23 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_569, clamp_max_13, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_569 = clamp_max_13 = primals_206 = None
    getitem_157: "f32[8, 906, 7, 7]" = convolution_backward_11[0]
    getitem_158: "f32[162, 906, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_4: "b8[8, 906, 7, 7]" = torch.ops.aten.le.Scalar(mul_388, 0.0)
    ge_2: "b8[8, 906, 7, 7]" = torch.ops.aten.ge.Scalar(mul_388, 6.0);  mul_388 = None
    bitwise_or_2: "b8[8, 906, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_4, ge_2);  le_4 = ge_2 = None
    where_4: "f32[8, 906, 7, 7]" = torch.ops.aten.where.self(bitwise_or_2, full_default_5, getitem_157);  bitwise_or_2 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_571: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, add_262);  add_262 = None
    mul_572: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sigmoid_24);  where_4 = None
    sum_24: "f32[8, 906, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2, 3], True);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_107: "f32[8, 906, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_24)
    mul_573: "f32[8, 906, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_24, sub_107);  sigmoid_24 = sub_107 = None
    mul_574: "f32[8, 906, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, mul_573);  sum_24 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_574, relu_10, primals_204, [906], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_574 = primals_204 = None
    getitem_160: "f32[8, 75, 1, 1]" = convolution_backward_12[0]
    getitem_161: "f32[906, 75, 1, 1]" = convolution_backward_12[1]
    getitem_162: "f32[906]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_5: "b8[8, 75, 1, 1]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_5: "f32[8, 75, 1, 1]" = torch.ops.aten.where.self(le_5, full_default_5, getitem_160);  le_5 = getitem_160 = None
    unsqueeze_368: "f32[1, 75]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
    unsqueeze_369: "f32[1, 75, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_25: "f32[75]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_108: "f32[8, 75, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_370);  convolution_61 = unsqueeze_370 = None
    mul_575: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(where_5, sub_108)
    sum_26: "f32[75]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_576: "f32[75]" = torch.ops.aten.mul.Tensor(sum_25, 0.125)
    unsqueeze_371: "f32[1, 75]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_372: "f32[1, 75, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_577: "f32[75]" = torch.ops.aten.mul.Tensor(sum_26, 0.125)
    mul_578: "f32[75]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_579: "f32[75]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_374: "f32[1, 75]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_375: "f32[1, 75, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_580: "f32[75]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_202);  primals_202 = None
    unsqueeze_377: "f32[1, 75]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_378: "f32[1, 75, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 75, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_581: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_376);  sub_108 = unsqueeze_376 = None
    sub_110: "f32[8, 75, 1, 1]" = torch.ops.aten.sub.Tensor(where_5, mul_581);  where_5 = mul_581 = None
    sub_111: "f32[8, 75, 1, 1]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_373);  sub_110 = unsqueeze_373 = None
    mul_582: "f32[8, 75, 1, 1]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_379);  sub_111 = unsqueeze_379 = None
    mul_583: "f32[75]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_154);  sum_26 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_582, mean_10, primals_200, [75], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_582 = mean_10 = primals_200 = None
    getitem_163: "f32[8, 906, 1, 1]" = convolution_backward_13[0]
    getitem_164: "f32[75, 906, 1, 1]" = convolution_backward_13[1]
    getitem_165: "f32[75]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 906, 7, 7]" = torch.ops.aten.expand.default(getitem_163, [8, 906, 7, 7]);  getitem_163 = None
    div_3: "f32[8, 906, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_331: "f32[8, 906, 7, 7]" = torch.ops.aten.add.Tensor(mul_572, div_3);  mul_572 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[906]" = torch.ops.aten.sum.dim_IntList(add_331, [0, 2, 3])
    sub_112: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_382);  convolution_60 = unsqueeze_382 = None
    mul_584: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(add_331, sub_112)
    sum_28: "f32[906]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_585: "f32[906]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    unsqueeze_383: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_384: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_586: "f32[906]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    mul_587: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_588: "f32[906]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_386: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_387: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_589: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_81);  primals_81 = None
    unsqueeze_389: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_390: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_590: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_388);  sub_112 = unsqueeze_388 = None
    sub_114: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(add_331, mul_590);  add_331 = mul_590 = None
    sub_115: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_385);  sub_114 = unsqueeze_385 = None
    mul_591: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_391);  sub_115 = unsqueeze_391 = None
    mul_592: "f32[906]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_151);  sum_28 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_591, mul_373, primals_199, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 906, [True, True, False]);  mul_591 = mul_373 = primals_199 = None
    getitem_166: "f32[8, 906, 7, 7]" = convolution_backward_14[0]
    getitem_167: "f32[906, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_595: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_166, mul_594);  getitem_166 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_29: "f32[906]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 2, 3])
    sub_117: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_394);  convolution_59 = unsqueeze_394 = None
    mul_596: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(mul_595, sub_117)
    sum_30: "f32[906]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 2, 3]);  mul_596 = None
    mul_597: "f32[906]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    unsqueeze_395: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_396: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_598: "f32[906]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    mul_599: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_600: "f32[906]" = torch.ops.aten.mul.Tensor(mul_598, mul_599);  mul_598 = mul_599 = None
    unsqueeze_398: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_399: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_601: "f32[906]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_79);  primals_79 = None
    unsqueeze_401: "f32[1, 906]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_402: "f32[1, 906, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 906, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_602: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_400);  sub_117 = unsqueeze_400 = None
    sub_119: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(mul_595, mul_602);  mul_595 = mul_602 = None
    sub_120: "f32[8, 906, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_397);  sub_119 = unsqueeze_397 = None
    mul_603: "f32[8, 906, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_403);  sub_120 = unsqueeze_403 = None
    mul_604: "f32[906]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_148);  sum_30 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_603, cat_7, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = cat_7 = primals_198 = None
    getitem_169: "f32[8, 151, 7, 7]" = convolution_backward_15[0]
    getitem_170: "f32[906, 151, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_333: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(slice_49, getitem_169);  slice_49 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_51: "f32[8, 140, 7, 7]" = torch.ops.aten.slice.Tensor(add_333, 1, 0, 140)
    slice_52: "f32[8, 11, 7, 7]" = torch.ops.aten.slice.Tensor(add_333, 1, 140, 151);  add_333 = None
    full_default_22: "f32[8, 151, 7, 7]" = torch.ops.aten.full.default([8, 151, 7, 7], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[8, 151, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_52, 1, 140, 9223372036854775807);  slice_52 = None
    slice_scatter_14: "f32[8, 151, 7, 7]" = torch.ops.aten.slice_scatter.default(full_default_22, slice_51, 1, 0, 140);  full_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_334: "f32[8, 151, 7, 7]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_14);  slice_scatter_12 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_31: "f32[151]" = torch.ops.aten.sum.dim_IntList(add_334, [0, 2, 3])
    sub_121: "f32[8, 151, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_406);  convolution_58 = unsqueeze_406 = None
    mul_605: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(add_334, sub_121)
    sum_32: "f32[151]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 2, 3]);  mul_605 = None
    mul_606: "f32[151]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    unsqueeze_407: "f32[1, 151]" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
    unsqueeze_408: "f32[1, 151, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_607: "f32[151]" = torch.ops.aten.mul.Tensor(sum_32, 0.002551020408163265)
    mul_608: "f32[151]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_609: "f32[151]" = torch.ops.aten.mul.Tensor(mul_607, mul_608);  mul_607 = mul_608 = None
    unsqueeze_410: "f32[1, 151]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_411: "f32[1, 151, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_610: "f32[151]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_77);  primals_77 = None
    unsqueeze_413: "f32[1, 151]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_414: "f32[1, 151, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 151, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_611: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_412);  sub_121 = unsqueeze_412 = None
    sub_123: "f32[8, 151, 7, 7]" = torch.ops.aten.sub.Tensor(add_334, mul_611);  add_334 = mul_611 = None
    sub_124: "f32[8, 151, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_409);  sub_123 = unsqueeze_409 = None
    mul_612: "f32[8, 151, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_415);  sub_124 = unsqueeze_415 = None
    mul_613: "f32[151]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_145);  sum_32 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_612, clamp_max_12, primals_197, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_612 = clamp_max_12 = primals_197 = None
    getitem_172: "f32[8, 840, 7, 7]" = convolution_backward_16[0]
    getitem_173: "f32[151, 840, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_6: "b8[8, 840, 7, 7]" = torch.ops.aten.le.Scalar(mul_358, 0.0)
    ge_3: "b8[8, 840, 7, 7]" = torch.ops.aten.ge.Scalar(mul_358, 6.0);  mul_358 = None
    bitwise_or_3: "b8[8, 840, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_6, ge_3);  le_6 = ge_3 = None
    where_6: "f32[8, 840, 7, 7]" = torch.ops.aten.where.self(bitwise_or_3, full_default_5, getitem_172);  bitwise_or_3 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_614: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, add_241);  add_241 = None
    mul_615: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sigmoid_22);  where_6 = None
    sum_33: "f32[8, 840, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [2, 3], True);  mul_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_125: "f32[8, 840, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_22)
    mul_616: "f32[8, 840, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_22, sub_125);  sigmoid_22 = sub_125 = None
    mul_617: "f32[8, 840, 1, 1]" = torch.ops.aten.mul.Tensor(sum_33, mul_616);  sum_33 = mul_616 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_617, relu_9, primals_195, [840], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_617 = primals_195 = None
    getitem_175: "f32[8, 70, 1, 1]" = convolution_backward_17[0]
    getitem_176: "f32[840, 70, 1, 1]" = convolution_backward_17[1]
    getitem_177: "f32[840]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_7: "b8[8, 70, 1, 1]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_7: "f32[8, 70, 1, 1]" = torch.ops.aten.where.self(le_7, full_default_5, getitem_175);  le_7 = getitem_175 = None
    unsqueeze_416: "f32[1, 70]" = torch.ops.aten.unsqueeze.default(squeeze_141, 0);  squeeze_141 = None
    unsqueeze_417: "f32[1, 70, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_34: "f32[70]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_126: "f32[8, 70, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_418);  convolution_56 = unsqueeze_418 = None
    mul_618: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(where_7, sub_126)
    sum_35: "f32[70]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3]);  mul_618 = None
    mul_619: "f32[70]" = torch.ops.aten.mul.Tensor(sum_34, 0.125)
    unsqueeze_419: "f32[1, 70]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
    unsqueeze_420: "f32[1, 70, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_620: "f32[70]" = torch.ops.aten.mul.Tensor(sum_35, 0.125)
    mul_621: "f32[70]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_622: "f32[70]" = torch.ops.aten.mul.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    unsqueeze_422: "f32[1, 70]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_423: "f32[1, 70, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_623: "f32[70]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_193);  primals_193 = None
    unsqueeze_425: "f32[1, 70]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_426: "f32[1, 70, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 70, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_624: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_424);  sub_126 = unsqueeze_424 = None
    sub_128: "f32[8, 70, 1, 1]" = torch.ops.aten.sub.Tensor(where_7, mul_624);  where_7 = mul_624 = None
    sub_129: "f32[8, 70, 1, 1]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_421);  sub_128 = unsqueeze_421 = None
    mul_625: "f32[8, 70, 1, 1]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_427);  sub_129 = unsqueeze_427 = None
    mul_626: "f32[70]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_142);  sum_35 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_625, mean_9, primals_191, [70], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_625 = mean_9 = primals_191 = None
    getitem_178: "f32[8, 840, 1, 1]" = convolution_backward_18[0]
    getitem_179: "f32[70, 840, 1, 1]" = convolution_backward_18[1]
    getitem_180: "f32[70]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 840, 7, 7]" = torch.ops.aten.expand.default(getitem_178, [8, 840, 7, 7]);  getitem_178 = None
    div_4: "f32[8, 840, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_335: "f32[8, 840, 7, 7]" = torch.ops.aten.add.Tensor(mul_615, div_4);  mul_615 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[840]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 2, 3])
    sub_130: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_430);  convolution_55 = unsqueeze_430 = None
    mul_627: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(add_335, sub_130)
    sum_37: "f32[840]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_628: "f32[840]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    unsqueeze_431: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_432: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_629: "f32[840]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    mul_630: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_631: "f32[840]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_434: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_435: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_632: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_75);  primals_75 = None
    unsqueeze_437: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_438: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_633: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_436);  sub_130 = unsqueeze_436 = None
    sub_132: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(add_335, mul_633);  add_335 = mul_633 = None
    sub_133: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_433);  sub_132 = unsqueeze_433 = None
    mul_634: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_439);  sub_133 = unsqueeze_439 = None
    mul_635: "f32[840]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_139);  sum_37 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_634, mul_343, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 840, [True, True, False]);  mul_634 = mul_343 = primals_190 = None
    getitem_181: "f32[8, 840, 7, 7]" = convolution_backward_19[0]
    getitem_182: "f32[840, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_638: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_181, mul_637);  getitem_181 = mul_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[840]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3])
    sub_135: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_442);  convolution_54 = unsqueeze_442 = None
    mul_639: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(mul_638, sub_135)
    sum_39: "f32[840]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 2, 3]);  mul_639 = None
    mul_640: "f32[840]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    unsqueeze_443: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_444: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_641: "f32[840]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    mul_642: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_643: "f32[840]" = torch.ops.aten.mul.Tensor(mul_641, mul_642);  mul_641 = mul_642 = None
    unsqueeze_446: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_447: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_644: "f32[840]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_73);  primals_73 = None
    unsqueeze_449: "f32[1, 840]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_450: "f32[1, 840, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 840, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_645: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_448);  sub_135 = unsqueeze_448 = None
    sub_137: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(mul_638, mul_645);  mul_638 = mul_645 = None
    sub_138: "f32[8, 840, 7, 7]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_445);  sub_137 = unsqueeze_445 = None
    mul_646: "f32[8, 840, 7, 7]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_451);  sub_138 = unsqueeze_451 = None
    mul_647: "f32[840]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_136);  sum_39 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_646, add_231, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_646 = add_231 = primals_189 = None
    getitem_184: "f32[8, 140, 7, 7]" = convolution_backward_20[0]
    getitem_185: "f32[840, 140, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_337: "f32[8, 140, 7, 7]" = torch.ops.aten.add.Tensor(slice_51, getitem_184);  slice_51 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[140]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 2, 3])
    sub_139: "f32[8, 140, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_454);  convolution_53 = unsqueeze_454 = None
    mul_648: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(add_337, sub_139)
    sum_41: "f32[140]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 2, 3]);  mul_648 = None
    mul_649: "f32[140]" = torch.ops.aten.mul.Tensor(sum_40, 0.002551020408163265)
    unsqueeze_455: "f32[1, 140]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_456: "f32[1, 140, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_650: "f32[140]" = torch.ops.aten.mul.Tensor(sum_41, 0.002551020408163265)
    mul_651: "f32[140]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_652: "f32[140]" = torch.ops.aten.mul.Tensor(mul_650, mul_651);  mul_650 = mul_651 = None
    unsqueeze_458: "f32[1, 140]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_459: "f32[1, 140, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_653: "f32[140]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_71);  primals_71 = None
    unsqueeze_461: "f32[1, 140]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_462: "f32[1, 140, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 140, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_654: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_460);  sub_139 = unsqueeze_460 = None
    sub_141: "f32[8, 140, 7, 7]" = torch.ops.aten.sub.Tensor(add_337, mul_654);  add_337 = mul_654 = None
    sub_142: "f32[8, 140, 7, 7]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_457);  sub_141 = unsqueeze_457 = None
    mul_655: "f32[8, 140, 7, 7]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_463);  sub_142 = unsqueeze_463 = None
    mul_656: "f32[140]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_133);  sum_41 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_655, clamp_max_11, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_655 = clamp_max_11 = primals_188 = None
    getitem_187: "f32[8, 768, 7, 7]" = convolution_backward_21[0]
    getitem_188: "f32[140, 768, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_8: "b8[8, 768, 7, 7]" = torch.ops.aten.le.Scalar(mul_328, 0.0)
    ge_4: "b8[8, 768, 7, 7]" = torch.ops.aten.ge.Scalar(mul_328, 6.0);  mul_328 = None
    bitwise_or_4: "b8[8, 768, 7, 7]" = torch.ops.aten.bitwise_or.Tensor(le_8, ge_4);  le_8 = ge_4 = None
    where_8: "f32[8, 768, 7, 7]" = torch.ops.aten.where.self(bitwise_or_4, full_default_5, getitem_187);  bitwise_or_4 = getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_657: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, add_221);  add_221 = None
    mul_658: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sigmoid_20);  where_8 = None
    sum_42: "f32[8, 768, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2, 3], True);  mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_143: "f32[8, 768, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_20)
    mul_659: "f32[8, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_20, sub_143);  sigmoid_20 = sub_143 = None
    mul_660: "f32[8, 768, 1, 1]" = torch.ops.aten.mul.Tensor(sum_42, mul_659);  sum_42 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_660, relu_8, primals_186, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_660 = primals_186 = None
    getitem_190: "f32[8, 64, 1, 1]" = convolution_backward_22[0]
    getitem_191: "f32[768, 64, 1, 1]" = convolution_backward_22[1]
    getitem_192: "f32[768]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_9: "b8[8, 64, 1, 1]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_9: "f32[8, 64, 1, 1]" = torch.ops.aten.where.self(le_9, full_default_5, getitem_190);  le_9 = getitem_190 = None
    unsqueeze_464: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
    unsqueeze_465: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_43: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_144: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_466);  convolution_51 = unsqueeze_466 = None
    mul_661: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(where_9, sub_144)
    sum_44: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_662: "f32[64]" = torch.ops.aten.mul.Tensor(sum_43, 0.125)
    unsqueeze_467: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_468: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_663: "f32[64]" = torch.ops.aten.mul.Tensor(sum_44, 0.125)
    mul_664: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_665: "f32[64]" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    unsqueeze_470: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_471: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_666: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_184);  primals_184 = None
    unsqueeze_473: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_474: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_667: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_472);  sub_144 = unsqueeze_472 = None
    sub_146: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(where_9, mul_667);  where_9 = mul_667 = None
    sub_147: "f32[8, 64, 1, 1]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_469);  sub_146 = unsqueeze_469 = None
    mul_668: "f32[8, 64, 1, 1]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_475);  sub_147 = unsqueeze_475 = None
    mul_669: "f32[64]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_130);  sum_44 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_668, mean_8, primals_182, [64], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_668 = mean_8 = primals_182 = None
    getitem_193: "f32[8, 768, 1, 1]" = convolution_backward_23[0]
    getitem_194: "f32[64, 768, 1, 1]" = convolution_backward_23[1]
    getitem_195: "f32[64]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 768, 7, 7]" = torch.ops.aten.expand.default(getitem_193, [8, 768, 7, 7]);  getitem_193 = None
    div_5: "f32[8, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_338: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_658, div_5);  mul_658 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 2, 3])
    sub_148: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_478);  convolution_50 = unsqueeze_478 = None
    mul_670: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_338, sub_148)
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
    mul_671: "f32[768]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    unsqueeze_479: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_480: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_672: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    mul_673: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_674: "f32[768]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    unsqueeze_482: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_483: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_675: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_69);  primals_69 = None
    unsqueeze_485: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_486: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_676: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_484);  sub_148 = unsqueeze_484 = None
    sub_150: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_338, mul_676);  add_338 = mul_676 = None
    sub_151: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_481);  sub_150 = unsqueeze_481 = None
    mul_677: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_487);  sub_151 = unsqueeze_487 = None
    mul_678: "f32[768]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_127);  sum_46 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_677, mul_313, primals_181, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 768, [True, True, False]);  mul_677 = mul_313 = primals_181 = None
    getitem_196: "f32[8, 768, 14, 14]" = convolution_backward_24[0]
    getitem_197: "f32[768, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_681: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_196, mul_680);  getitem_196 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3])
    sub_153: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_490);  convolution_49 = unsqueeze_490 = None
    mul_682: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_681, sub_153)
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
    mul_683: "f32[768]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    unsqueeze_491: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_492: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_684: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    mul_685: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_686: "f32[768]" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
    unsqueeze_494: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_495: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_687: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_67);  primals_67 = None
    unsqueeze_497: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
    unsqueeze_498: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_688: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_496);  sub_153 = unsqueeze_496 = None
    sub_155: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(mul_681, mul_688);  mul_681 = mul_688 = None
    sub_156: "f32[8, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_493);  sub_155 = unsqueeze_493 = None
    mul_689: "f32[8, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_499);  sub_156 = unsqueeze_499 = None
    mul_690: "f32[768]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_124);  sum_48 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_689, cat_6, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_689 = cat_6 = primals_180 = None
    getitem_199: "f32[8, 128, 14, 14]" = convolution_backward_25[0]
    getitem_200: "f32[768, 128, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_53: "f32[8, 117, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_199, 1, 0, 117)
    slice_54: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_199, 1, 117, 128);  getitem_199 = None
    full_default_32: "f32[8, 128, 14, 14]" = torch.ops.aten.full.default([8, 128, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[8, 128, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_32, slice_54, 1, 117, 9223372036854775807);  slice_54 = None
    slice_scatter_18: "f32[8, 128, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_32, slice_53, 1, 0, 117);  full_default_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_340: "f32[8, 128, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_18);  slice_scatter_16 = slice_scatter_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_49: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 2, 3])
    sub_157: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_502);  convolution_48 = unsqueeze_502 = None
    mul_691: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(add_340, sub_157)
    sum_50: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_692: "f32[128]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    unsqueeze_503: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_504: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_693: "f32[128]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    mul_694: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_695: "f32[128]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_506: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_507: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_696: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_65);  primals_65 = None
    unsqueeze_509: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_510: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_697: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_508);  sub_157 = unsqueeze_508 = None
    sub_159: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(add_340, mul_697);  add_340 = mul_697 = None
    sub_160: "f32[8, 128, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_505);  sub_159 = unsqueeze_505 = None
    mul_698: "f32[8, 128, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_511);  sub_160 = unsqueeze_511 = None
    mul_699: "f32[128]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_121);  sum_50 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_698, clamp_max_10, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_698 = clamp_max_10 = primals_179 = None
    getitem_202: "f32[8, 702, 14, 14]" = convolution_backward_26[0]
    getitem_203: "f32[128, 702, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_10: "b8[8, 702, 14, 14]" = torch.ops.aten.le.Scalar(mul_298, 0.0)
    ge_5: "b8[8, 702, 14, 14]" = torch.ops.aten.ge.Scalar(mul_298, 6.0);  mul_298 = None
    bitwise_or_5: "b8[8, 702, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_10, ge_5);  le_10 = ge_5 = None
    where_10: "f32[8, 702, 14, 14]" = torch.ops.aten.where.self(bitwise_or_5, full_default_5, getitem_202);  bitwise_or_5 = getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_700: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, add_200);  add_200 = None
    mul_701: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sigmoid_18);  where_10 = None
    sum_51: "f32[8, 702, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [2, 3], True);  mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_161: "f32[8, 702, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_18)
    mul_702: "f32[8, 702, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_18, sub_161);  sigmoid_18 = sub_161 = None
    mul_703: "f32[8, 702, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, mul_702);  sum_51 = mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_703, relu_7, primals_177, [702], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_703 = primals_177 = None
    getitem_205: "f32[8, 58, 1, 1]" = convolution_backward_27[0]
    getitem_206: "f32[702, 58, 1, 1]" = convolution_backward_27[1]
    getitem_207: "f32[702]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_11: "b8[8, 58, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_11: "f32[8, 58, 1, 1]" = torch.ops.aten.where.self(le_11, full_default_5, getitem_205);  le_11 = getitem_205 = None
    unsqueeze_512: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
    unsqueeze_513: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_52: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_162: "f32[8, 58, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_514);  convolution_46 = unsqueeze_514 = None
    mul_704: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(where_11, sub_162)
    sum_53: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3]);  mul_704 = None
    mul_705: "f32[58]" = torch.ops.aten.mul.Tensor(sum_52, 0.125)
    unsqueeze_515: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
    unsqueeze_516: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_706: "f32[58]" = torch.ops.aten.mul.Tensor(sum_53, 0.125)
    mul_707: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_708: "f32[58]" = torch.ops.aten.mul.Tensor(mul_706, mul_707);  mul_706 = mul_707 = None
    unsqueeze_518: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_519: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_709: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_175);  primals_175 = None
    unsqueeze_521: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_522: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_710: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_520);  sub_162 = unsqueeze_520 = None
    sub_164: "f32[8, 58, 1, 1]" = torch.ops.aten.sub.Tensor(where_11, mul_710);  where_11 = mul_710 = None
    sub_165: "f32[8, 58, 1, 1]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_517);  sub_164 = unsqueeze_517 = None
    mul_711: "f32[8, 58, 1, 1]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_523);  sub_165 = unsqueeze_523 = None
    mul_712: "f32[58]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_118);  sum_53 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_711, mean_7, primals_173, [58], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_711 = mean_7 = primals_173 = None
    getitem_208: "f32[8, 702, 1, 1]" = convolution_backward_28[0]
    getitem_209: "f32[58, 702, 1, 1]" = convolution_backward_28[1]
    getitem_210: "f32[58]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 702, 14, 14]" = torch.ops.aten.expand.default(getitem_208, [8, 702, 14, 14]);  getitem_208 = None
    div_6: "f32[8, 702, 14, 14]" = torch.ops.aten.div.Scalar(expand_6, 196);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_341: "f32[8, 702, 14, 14]" = torch.ops.aten.add.Tensor(mul_701, div_6);  mul_701 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[702]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 2, 3])
    sub_166: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_526);  convolution_45 = unsqueeze_526 = None
    mul_713: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(add_341, sub_166)
    sum_55: "f32[702]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 2, 3]);  mul_713 = None
    mul_714: "f32[702]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_528: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_715: "f32[702]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_716: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_717: "f32[702]" = torch.ops.aten.mul.Tensor(mul_715, mul_716);  mul_715 = mul_716 = None
    unsqueeze_530: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_531: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_718: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_63);  primals_63 = None
    unsqueeze_533: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_534: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_719: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_532);  sub_166 = unsqueeze_532 = None
    sub_168: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(add_341, mul_719);  add_341 = mul_719 = None
    sub_169: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_529);  sub_168 = unsqueeze_529 = None
    mul_720: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_535);  sub_169 = unsqueeze_535 = None
    mul_721: "f32[702]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_115);  sum_55 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_720, mul_283, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 702, [True, True, False]);  mul_720 = mul_283 = primals_172 = None
    getitem_211: "f32[8, 702, 14, 14]" = convolution_backward_29[0]
    getitem_212: "f32[702, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_724: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, mul_723);  getitem_211 = mul_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[702]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3])
    sub_171: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_538);  convolution_44 = unsqueeze_538 = None
    mul_725: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(mul_724, sub_171)
    sum_57: "f32[702]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_726: "f32[702]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_540: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_727: "f32[702]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_728: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_729: "f32[702]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_542: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_543: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_730: "f32[702]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_61);  primals_61 = None
    unsqueeze_545: "f32[1, 702]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_546: "f32[1, 702, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 702, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_731: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_544);  sub_171 = unsqueeze_544 = None
    sub_173: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(mul_724, mul_731);  mul_724 = mul_731 = None
    sub_174: "f32[8, 702, 14, 14]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_541);  sub_173 = unsqueeze_541 = None
    mul_732: "f32[8, 702, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_547);  sub_174 = unsqueeze_547 = None
    mul_733: "f32[702]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_112);  sum_57 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_732, cat_5, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_732 = cat_5 = primals_171 = None
    getitem_214: "f32[8, 117, 14, 14]" = convolution_backward_30[0]
    getitem_215: "f32[702, 117, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_343: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(slice_53, getitem_214);  slice_53 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_55: "f32[8, 106, 14, 14]" = torch.ops.aten.slice.Tensor(add_343, 1, 0, 106)
    slice_56: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_343, 1, 106, 117);  add_343 = None
    full_default_39: "f32[8, 117, 14, 14]" = torch.ops.aten.full.default([8, 117, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_20: "f32[8, 117, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_39, slice_56, 1, 106, 9223372036854775807);  slice_56 = None
    slice_scatter_22: "f32[8, 117, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_39, slice_55, 1, 0, 106);  full_default_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_344: "f32[8, 117, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_22);  slice_scatter_20 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[117]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 2, 3])
    sub_175: "f32[8, 117, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_550);  convolution_43 = unsqueeze_550 = None
    mul_734: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(add_344, sub_175)
    sum_59: "f32[117]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
    mul_735: "f32[117]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 117]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_552: "f32[1, 117, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_736: "f32[117]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_737: "f32[117]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_738: "f32[117]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    unsqueeze_554: "f32[1, 117]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_555: "f32[1, 117, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_739: "f32[117]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_59);  primals_59 = None
    unsqueeze_557: "f32[1, 117]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_558: "f32[1, 117, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 117, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_740: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_556);  sub_175 = unsqueeze_556 = None
    sub_177: "f32[8, 117, 14, 14]" = torch.ops.aten.sub.Tensor(add_344, mul_740);  add_344 = mul_740 = None
    sub_178: "f32[8, 117, 14, 14]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_553);  sub_177 = unsqueeze_553 = None
    mul_741: "f32[8, 117, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_559);  sub_178 = unsqueeze_559 = None
    mul_742: "f32[117]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_109);  sum_59 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_741, clamp_max_9, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_741 = clamp_max_9 = primals_170 = None
    getitem_217: "f32[8, 636, 14, 14]" = convolution_backward_31[0]
    getitem_218: "f32[117, 636, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_12: "b8[8, 636, 14, 14]" = torch.ops.aten.le.Scalar(mul_268, 0.0)
    ge_6: "b8[8, 636, 14, 14]" = torch.ops.aten.ge.Scalar(mul_268, 6.0);  mul_268 = None
    bitwise_or_6: "b8[8, 636, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_12, ge_6);  le_12 = ge_6 = None
    where_12: "f32[8, 636, 14, 14]" = torch.ops.aten.where.self(bitwise_or_6, full_default_5, getitem_217);  bitwise_or_6 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_743: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, add_179);  add_179 = None
    mul_744: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sigmoid_16);  where_12 = None
    sum_60: "f32[8, 636, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [2, 3], True);  mul_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_179: "f32[8, 636, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_16)
    mul_745: "f32[8, 636, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_16, sub_179);  sigmoid_16 = sub_179 = None
    mul_746: "f32[8, 636, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_745);  sum_60 = mul_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_746, relu_6, primals_168, [636], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_746 = primals_168 = None
    getitem_220: "f32[8, 53, 1, 1]" = convolution_backward_32[0]
    getitem_221: "f32[636, 53, 1, 1]" = convolution_backward_32[1]
    getitem_222: "f32[636]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_13: "b8[8, 53, 1, 1]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_13: "f32[8, 53, 1, 1]" = torch.ops.aten.where.self(le_13, full_default_5, getitem_220);  le_13 = getitem_220 = None
    unsqueeze_560: "f32[1, 53]" = torch.ops.aten.unsqueeze.default(squeeze_105, 0);  squeeze_105 = None
    unsqueeze_561: "f32[1, 53, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_61: "f32[53]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_180: "f32[8, 53, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_562);  convolution_41 = unsqueeze_562 = None
    mul_747: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(where_13, sub_180)
    sum_62: "f32[53]" = torch.ops.aten.sum.dim_IntList(mul_747, [0, 2, 3]);  mul_747 = None
    mul_748: "f32[53]" = torch.ops.aten.mul.Tensor(sum_61, 0.125)
    unsqueeze_563: "f32[1, 53]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_564: "f32[1, 53, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_749: "f32[53]" = torch.ops.aten.mul.Tensor(sum_62, 0.125)
    mul_750: "f32[53]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_751: "f32[53]" = torch.ops.aten.mul.Tensor(mul_749, mul_750);  mul_749 = mul_750 = None
    unsqueeze_566: "f32[1, 53]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_567: "f32[1, 53, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_752: "f32[53]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_166);  primals_166 = None
    unsqueeze_569: "f32[1, 53]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_570: "f32[1, 53, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 53, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_753: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_568);  sub_180 = unsqueeze_568 = None
    sub_182: "f32[8, 53, 1, 1]" = torch.ops.aten.sub.Tensor(where_13, mul_753);  where_13 = mul_753 = None
    sub_183: "f32[8, 53, 1, 1]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_565);  sub_182 = unsqueeze_565 = None
    mul_754: "f32[8, 53, 1, 1]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_571);  sub_183 = unsqueeze_571 = None
    mul_755: "f32[53]" = torch.ops.aten.mul.Tensor(sum_62, squeeze_106);  sum_62 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_754, mean_6, primals_164, [53], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_754 = mean_6 = primals_164 = None
    getitem_223: "f32[8, 636, 1, 1]" = convolution_backward_33[0]
    getitem_224: "f32[53, 636, 1, 1]" = convolution_backward_33[1]
    getitem_225: "f32[53]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 636, 14, 14]" = torch.ops.aten.expand.default(getitem_223, [8, 636, 14, 14]);  getitem_223 = None
    div_7: "f32[8, 636, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_345: "f32[8, 636, 14, 14]" = torch.ops.aten.add.Tensor(mul_744, div_7);  mul_744 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_63: "f32[636]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 2, 3])
    sub_184: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_574);  convolution_40 = unsqueeze_574 = None
    mul_756: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(add_345, sub_184)
    sum_64: "f32[636]" = torch.ops.aten.sum.dim_IntList(mul_756, [0, 2, 3]);  mul_756 = None
    mul_757: "f32[636]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_576: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_758: "f32[636]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    mul_759: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_760: "f32[636]" = torch.ops.aten.mul.Tensor(mul_758, mul_759);  mul_758 = mul_759 = None
    unsqueeze_578: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_579: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_761: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_57);  primals_57 = None
    unsqueeze_581: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_582: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_762: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_580);  sub_184 = unsqueeze_580 = None
    sub_186: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(add_345, mul_762);  add_345 = mul_762 = None
    sub_187: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_577);  sub_186 = unsqueeze_577 = None
    mul_763: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_583);  sub_187 = unsqueeze_583 = None
    mul_764: "f32[636]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_103);  sum_64 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_763, mul_253, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 636, [True, True, False]);  mul_763 = mul_253 = primals_163 = None
    getitem_226: "f32[8, 636, 14, 14]" = convolution_backward_34[0]
    getitem_227: "f32[636, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_767: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_226, mul_766);  getitem_226 = mul_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[636]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3])
    sub_189: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_586);  convolution_39 = unsqueeze_586 = None
    mul_768: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(mul_767, sub_189)
    sum_66: "f32[636]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 2, 3]);  mul_768 = None
    mul_769: "f32[636]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_588: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_770: "f32[636]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_771: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_772: "f32[636]" = torch.ops.aten.mul.Tensor(mul_770, mul_771);  mul_770 = mul_771 = None
    unsqueeze_590: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_591: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_773: "f32[636]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_55);  primals_55 = None
    unsqueeze_593: "f32[1, 636]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_594: "f32[1, 636, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 636, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_774: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_592);  sub_189 = unsqueeze_592 = None
    sub_191: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(mul_767, mul_774);  mul_767 = mul_774 = None
    sub_192: "f32[8, 636, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_589);  sub_191 = unsqueeze_589 = None
    mul_775: "f32[8, 636, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_595);  sub_192 = unsqueeze_595 = None
    mul_776: "f32[636]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_100);  sum_66 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_775, cat_4, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_775 = cat_4 = primals_162 = None
    getitem_229: "f32[8, 106, 14, 14]" = convolution_backward_35[0]
    getitem_230: "f32[636, 106, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_347: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(slice_55, getitem_229);  slice_55 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_57: "f32[8, 95, 14, 14]" = torch.ops.aten.slice.Tensor(add_347, 1, 0, 95)
    slice_58: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_347, 1, 95, 106);  add_347 = None
    full_default_46: "f32[8, 106, 14, 14]" = torch.ops.aten.full.default([8, 106, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_24: "f32[8, 106, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_46, slice_58, 1, 95, 9223372036854775807);  slice_58 = None
    slice_scatter_26: "f32[8, 106, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_46, slice_57, 1, 0, 95);  full_default_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_348: "f32[8, 106, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_26);  slice_scatter_24 = slice_scatter_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[106]" = torch.ops.aten.sum.dim_IntList(add_348, [0, 2, 3])
    sub_193: "f32[8, 106, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_598);  convolution_38 = unsqueeze_598 = None
    mul_777: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, sub_193)
    sum_68: "f32[106]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
    mul_778: "f32[106]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 106]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_600: "f32[1, 106, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_779: "f32[106]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_780: "f32[106]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_781: "f32[106]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
    unsqueeze_602: "f32[1, 106]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_603: "f32[1, 106, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_782: "f32[106]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_53);  primals_53 = None
    unsqueeze_605: "f32[1, 106]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_606: "f32[1, 106, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 106, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_783: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_604);  sub_193 = unsqueeze_604 = None
    sub_195: "f32[8, 106, 14, 14]" = torch.ops.aten.sub.Tensor(add_348, mul_783);  add_348 = mul_783 = None
    sub_196: "f32[8, 106, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_601);  sub_195 = unsqueeze_601 = None
    mul_784: "f32[8, 106, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_607);  sub_196 = unsqueeze_607 = None
    mul_785: "f32[106]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_97);  sum_68 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_784, clamp_max_8, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = clamp_max_8 = primals_161 = None
    getitem_232: "f32[8, 570, 14, 14]" = convolution_backward_36[0]
    getitem_233: "f32[106, 570, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_14: "b8[8, 570, 14, 14]" = torch.ops.aten.le.Scalar(mul_238, 0.0)
    ge_7: "b8[8, 570, 14, 14]" = torch.ops.aten.ge.Scalar(mul_238, 6.0);  mul_238 = None
    bitwise_or_7: "b8[8, 570, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_14, ge_7);  le_14 = ge_7 = None
    where_14: "f32[8, 570, 14, 14]" = torch.ops.aten.where.self(bitwise_or_7, full_default_5, getitem_232);  bitwise_or_7 = getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_786: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, add_158);  add_158 = None
    mul_787: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sigmoid_14);  where_14 = None
    sum_69: "f32[8, 570, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_786, [2, 3], True);  mul_786 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_197: "f32[8, 570, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_14)
    mul_788: "f32[8, 570, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_14, sub_197);  sigmoid_14 = sub_197 = None
    mul_789: "f32[8, 570, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_788);  sum_69 = mul_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_789, relu_5, primals_159, [570], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_789 = primals_159 = None
    getitem_235: "f32[8, 47, 1, 1]" = convolution_backward_37[0]
    getitem_236: "f32[570, 47, 1, 1]" = convolution_backward_37[1]
    getitem_237: "f32[570]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_15: "b8[8, 47, 1, 1]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_15: "f32[8, 47, 1, 1]" = torch.ops.aten.where.self(le_15, full_default_5, getitem_235);  le_15 = getitem_235 = None
    unsqueeze_608: "f32[1, 47]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
    unsqueeze_609: "f32[1, 47, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_70: "f32[47]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_198: "f32[8, 47, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_610);  convolution_36 = unsqueeze_610 = None
    mul_790: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(where_15, sub_198)
    sum_71: "f32[47]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 2, 3]);  mul_790 = None
    mul_791: "f32[47]" = torch.ops.aten.mul.Tensor(sum_70, 0.125)
    unsqueeze_611: "f32[1, 47]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_612: "f32[1, 47, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_792: "f32[47]" = torch.ops.aten.mul.Tensor(sum_71, 0.125)
    mul_793: "f32[47]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_794: "f32[47]" = torch.ops.aten.mul.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    unsqueeze_614: "f32[1, 47]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_615: "f32[1, 47, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_795: "f32[47]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_157);  primals_157 = None
    unsqueeze_617: "f32[1, 47]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_618: "f32[1, 47, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 47, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_796: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_616);  sub_198 = unsqueeze_616 = None
    sub_200: "f32[8, 47, 1, 1]" = torch.ops.aten.sub.Tensor(where_15, mul_796);  where_15 = mul_796 = None
    sub_201: "f32[8, 47, 1, 1]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_613);  sub_200 = unsqueeze_613 = None
    mul_797: "f32[8, 47, 1, 1]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_619);  sub_201 = unsqueeze_619 = None
    mul_798: "f32[47]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_94);  sum_71 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_797, mean_5, primals_155, [47], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_797 = mean_5 = primals_155 = None
    getitem_238: "f32[8, 570, 1, 1]" = convolution_backward_38[0]
    getitem_239: "f32[47, 570, 1, 1]" = convolution_backward_38[1]
    getitem_240: "f32[47]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 570, 14, 14]" = torch.ops.aten.expand.default(getitem_238, [8, 570, 14, 14]);  getitem_238 = None
    div_8: "f32[8, 570, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_349: "f32[8, 570, 14, 14]" = torch.ops.aten.add.Tensor(mul_787, div_8);  mul_787 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[570]" = torch.ops.aten.sum.dim_IntList(add_349, [0, 2, 3])
    sub_202: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_622);  convolution_35 = unsqueeze_622 = None
    mul_799: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(add_349, sub_202)
    sum_73: "f32[570]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2, 3]);  mul_799 = None
    mul_800: "f32[570]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_624: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_801: "f32[570]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_802: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_803: "f32[570]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_626: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_627: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_804: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_51);  primals_51 = None
    unsqueeze_629: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_630: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_805: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_628);  sub_202 = unsqueeze_628 = None
    sub_204: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(add_349, mul_805);  add_349 = mul_805 = None
    sub_205: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(sub_204, unsqueeze_625);  sub_204 = unsqueeze_625 = None
    mul_806: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_631);  sub_205 = unsqueeze_631 = None
    mul_807: "f32[570]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_91);  sum_73 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_806, mul_223, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 570, [True, True, False]);  mul_806 = mul_223 = primals_154 = None
    getitem_241: "f32[8, 570, 14, 14]" = convolution_backward_39[0]
    getitem_242: "f32[570, 1, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_810: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_241, mul_809);  getitem_241 = mul_809 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[570]" = torch.ops.aten.sum.dim_IntList(mul_810, [0, 2, 3])
    sub_207: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_634);  convolution_34 = unsqueeze_634 = None
    mul_811: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(mul_810, sub_207)
    sum_75: "f32[570]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 2, 3]);  mul_811 = None
    mul_812: "f32[570]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_636: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_813: "f32[570]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_814: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_815: "f32[570]" = torch.ops.aten.mul.Tensor(mul_813, mul_814);  mul_813 = mul_814 = None
    unsqueeze_638: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_639: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_816: "f32[570]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_49);  primals_49 = None
    unsqueeze_641: "f32[1, 570]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_642: "f32[1, 570, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 570, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_817: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_640);  sub_207 = unsqueeze_640 = None
    sub_209: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(mul_810, mul_817);  mul_810 = mul_817 = None
    sub_210: "f32[8, 570, 14, 14]" = torch.ops.aten.sub.Tensor(sub_209, unsqueeze_637);  sub_209 = unsqueeze_637 = None
    mul_818: "f32[8, 570, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_643);  sub_210 = unsqueeze_643 = None
    mul_819: "f32[570]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_88);  sum_75 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_818, cat_3, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_818 = cat_3 = primals_153 = None
    getitem_244: "f32[8, 95, 14, 14]" = convolution_backward_40[0]
    getitem_245: "f32[570, 95, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_351: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(slice_57, getitem_244);  slice_57 = getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_59: "f32[8, 84, 14, 14]" = torch.ops.aten.slice.Tensor(add_351, 1, 0, 84)
    slice_60: "f32[8, 11, 14, 14]" = torch.ops.aten.slice.Tensor(add_351, 1, 84, 95);  add_351 = None
    full_default_53: "f32[8, 95, 14, 14]" = torch.ops.aten.full.default([8, 95, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_28: "f32[8, 95, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_53, slice_60, 1, 84, 9223372036854775807);  slice_60 = None
    slice_scatter_30: "f32[8, 95, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_53, slice_59, 1, 0, 84);  full_default_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_352: "f32[8, 95, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_30);  slice_scatter_28 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[95]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 2, 3])
    sub_211: "f32[8, 95, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_646);  convolution_33 = unsqueeze_646 = None
    mul_820: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(add_352, sub_211)
    sum_77: "f32[95]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 2, 3]);  mul_820 = None
    mul_821: "f32[95]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 95]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_648: "f32[1, 95, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_822: "f32[95]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_823: "f32[95]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_824: "f32[95]" = torch.ops.aten.mul.Tensor(mul_822, mul_823);  mul_822 = mul_823 = None
    unsqueeze_650: "f32[1, 95]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_651: "f32[1, 95, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_825: "f32[95]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_47);  primals_47 = None
    unsqueeze_653: "f32[1, 95]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_654: "f32[1, 95, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 95, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_826: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_652);  sub_211 = unsqueeze_652 = None
    sub_213: "f32[8, 95, 14, 14]" = torch.ops.aten.sub.Tensor(add_352, mul_826);  add_352 = mul_826 = None
    sub_214: "f32[8, 95, 14, 14]" = torch.ops.aten.sub.Tensor(sub_213, unsqueeze_649);  sub_213 = unsqueeze_649 = None
    mul_827: "f32[8, 95, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_655);  sub_214 = unsqueeze_655 = None
    mul_828: "f32[95]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_85);  sum_77 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_827, clamp_max_7, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_827 = clamp_max_7 = primals_152 = None
    getitem_247: "f32[8, 504, 14, 14]" = convolution_backward_41[0]
    getitem_248: "f32[95, 504, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_16: "b8[8, 504, 14, 14]" = torch.ops.aten.le.Scalar(mul_208, 0.0)
    ge_8: "b8[8, 504, 14, 14]" = torch.ops.aten.ge.Scalar(mul_208, 6.0);  mul_208 = None
    bitwise_or_8: "b8[8, 504, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_16, ge_8);  le_16 = ge_8 = None
    where_16: "f32[8, 504, 14, 14]" = torch.ops.aten.where.self(bitwise_or_8, full_default_5, getitem_247);  bitwise_or_8 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_829: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, add_137);  add_137 = None
    mul_830: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sigmoid_12);  where_16 = None
    sum_78: "f32[8, 504, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_829, [2, 3], True);  mul_829 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_215: "f32[8, 504, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
    mul_831: "f32[8, 504, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_215);  sigmoid_12 = sub_215 = None
    mul_832: "f32[8, 504, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_831);  sum_78 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_832, relu_4, primals_150, [504], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_832 = primals_150 = None
    getitem_250: "f32[8, 42, 1, 1]" = convolution_backward_42[0]
    getitem_251: "f32[504, 42, 1, 1]" = convolution_backward_42[1]
    getitem_252: "f32[504]" = convolution_backward_42[2];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_17: "b8[8, 42, 1, 1]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_17: "f32[8, 42, 1, 1]" = torch.ops.aten.where.self(le_17, full_default_5, getitem_250);  le_17 = getitem_250 = None
    unsqueeze_656: "f32[1, 42]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
    unsqueeze_657: "f32[1, 42, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_79: "f32[42]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_216: "f32[8, 42, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_658);  convolution_31 = unsqueeze_658 = None
    mul_833: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(where_17, sub_216)
    sum_80: "f32[42]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
    mul_834: "f32[42]" = torch.ops.aten.mul.Tensor(sum_79, 0.125)
    unsqueeze_659: "f32[1, 42]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_660: "f32[1, 42, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_835: "f32[42]" = torch.ops.aten.mul.Tensor(sum_80, 0.125)
    mul_836: "f32[42]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_837: "f32[42]" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
    unsqueeze_662: "f32[1, 42]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_663: "f32[1, 42, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_838: "f32[42]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_148);  primals_148 = None
    unsqueeze_665: "f32[1, 42]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_666: "f32[1, 42, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 42, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_839: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_664);  sub_216 = unsqueeze_664 = None
    sub_218: "f32[8, 42, 1, 1]" = torch.ops.aten.sub.Tensor(where_17, mul_839);  where_17 = mul_839 = None
    sub_219: "f32[8, 42, 1, 1]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_661);  sub_218 = unsqueeze_661 = None
    mul_840: "f32[8, 42, 1, 1]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_667);  sub_219 = unsqueeze_667 = None
    mul_841: "f32[42]" = torch.ops.aten.mul.Tensor(sum_80, squeeze_82);  sum_80 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_840, mean_4, primals_146, [42], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_840 = mean_4 = primals_146 = None
    getitem_253: "f32[8, 504, 1, 1]" = convolution_backward_43[0]
    getitem_254: "f32[42, 504, 1, 1]" = convolution_backward_43[1]
    getitem_255: "f32[42]" = convolution_backward_43[2];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 504, 14, 14]" = torch.ops.aten.expand.default(getitem_253, [8, 504, 14, 14]);  getitem_253 = None
    div_9: "f32[8, 504, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_353: "f32[8, 504, 14, 14]" = torch.ops.aten.add.Tensor(mul_830, div_9);  mul_830 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[504]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 2, 3])
    sub_220: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_670);  convolution_30 = unsqueeze_670 = None
    mul_842: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(add_353, sub_220)
    sum_82: "f32[504]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
    mul_843: "f32[504]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_672: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_844: "f32[504]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_845: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_846: "f32[504]" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_674: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_675: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_847: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_45);  primals_45 = None
    unsqueeze_677: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_678: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_848: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_676);  sub_220 = unsqueeze_676 = None
    sub_222: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(add_353, mul_848);  add_353 = mul_848 = None
    sub_223: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_673);  sub_222 = unsqueeze_673 = None
    mul_849: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_679);  sub_223 = unsqueeze_679 = None
    mul_850: "f32[504]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_79);  sum_82 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_849, mul_193, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 504, [True, True, False]);  mul_849 = mul_193 = primals_145 = None
    getitem_256: "f32[8, 504, 14, 14]" = convolution_backward_44[0]
    getitem_257: "f32[504, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_853: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_256, mul_852);  getitem_256 = mul_852 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_83: "f32[504]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3])
    sub_225: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_682);  convolution_29 = unsqueeze_682 = None
    mul_854: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(mul_853, sub_225)
    sum_84: "f32[504]" = torch.ops.aten.sum.dim_IntList(mul_854, [0, 2, 3]);  mul_854 = None
    mul_855: "f32[504]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_684: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_856: "f32[504]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    mul_857: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_858: "f32[504]" = torch.ops.aten.mul.Tensor(mul_856, mul_857);  mul_856 = mul_857 = None
    unsqueeze_686: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_687: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_859: "f32[504]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_43);  primals_43 = None
    unsqueeze_689: "f32[1, 504]" = torch.ops.aten.unsqueeze.default(mul_859, 0);  mul_859 = None
    unsqueeze_690: "f32[1, 504, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 504, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_860: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_688);  sub_225 = unsqueeze_688 = None
    sub_227: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(mul_853, mul_860);  mul_853 = mul_860 = None
    sub_228: "f32[8, 504, 14, 14]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_685);  sub_227 = unsqueeze_685 = None
    mul_861: "f32[8, 504, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_691);  sub_228 = unsqueeze_691 = None
    mul_862: "f32[504]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_76);  sum_84 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_861, cat_2, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_861 = cat_2 = primals_144 = None
    getitem_259: "f32[8, 84, 14, 14]" = convolution_backward_45[0]
    getitem_260: "f32[504, 84, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_355: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(slice_59, getitem_259);  slice_59 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_61: "f32[8, 72, 14, 14]" = torch.ops.aten.slice.Tensor(add_355, 1, 0, 72)
    slice_62: "f32[8, 12, 14, 14]" = torch.ops.aten.slice.Tensor(add_355, 1, 72, 84);  add_355 = None
    full_default_60: "f32[8, 84, 14, 14]" = torch.ops.aten.full.default([8, 84, 14, 14], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_32: "f32[8, 84, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_60, slice_62, 1, 72, 9223372036854775807);  slice_62 = None
    slice_scatter_34: "f32[8, 84, 14, 14]" = torch.ops.aten.slice_scatter.default(full_default_60, slice_61, 1, 0, 72);  full_default_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_356: "f32[8, 84, 14, 14]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_34);  slice_scatter_32 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_85: "f32[84]" = torch.ops.aten.sum.dim_IntList(add_356, [0, 2, 3])
    sub_229: "f32[8, 84, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_694);  convolution_28 = unsqueeze_694 = None
    mul_863: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(add_356, sub_229)
    sum_86: "f32[84]" = torch.ops.aten.sum.dim_IntList(mul_863, [0, 2, 3]);  mul_863 = None
    mul_864: "f32[84]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 84]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_696: "f32[1, 84, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_865: "f32[84]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    mul_866: "f32[84]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_867: "f32[84]" = torch.ops.aten.mul.Tensor(mul_865, mul_866);  mul_865 = mul_866 = None
    unsqueeze_698: "f32[1, 84]" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
    unsqueeze_699: "f32[1, 84, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_868: "f32[84]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_41);  primals_41 = None
    unsqueeze_701: "f32[1, 84]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_702: "f32[1, 84, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 84, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_869: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_700);  sub_229 = unsqueeze_700 = None
    sub_231: "f32[8, 84, 14, 14]" = torch.ops.aten.sub.Tensor(add_356, mul_869);  add_356 = mul_869 = None
    sub_232: "f32[8, 84, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_697);  sub_231 = unsqueeze_697 = None
    mul_870: "f32[8, 84, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_703);  sub_232 = unsqueeze_703 = None
    mul_871: "f32[84]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_73);  sum_86 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_870, clamp_max_6, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_870 = clamp_max_6 = primals_143 = None
    getitem_262: "f32[8, 432, 14, 14]" = convolution_backward_46[0]
    getitem_263: "f32[84, 432, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_18: "b8[8, 432, 14, 14]" = torch.ops.aten.le.Scalar(mul_178, 0.0)
    ge_9: "b8[8, 432, 14, 14]" = torch.ops.aten.ge.Scalar(mul_178, 6.0);  mul_178 = None
    bitwise_or_9: "b8[8, 432, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_18, ge_9);  le_18 = ge_9 = None
    where_18: "f32[8, 432, 14, 14]" = torch.ops.aten.where.self(bitwise_or_9, full_default_5, getitem_262);  bitwise_or_9 = getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_872: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, add_116);  add_116 = None
    mul_873: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sigmoid_10);  where_18 = None
    sum_87: "f32[8, 432, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_872, [2, 3], True);  mul_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_233: "f32[8, 432, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
    mul_874: "f32[8, 432, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_233);  sigmoid_10 = sub_233 = None
    mul_875: "f32[8, 432, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_874);  sum_87 = mul_874 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_875, relu_3, primals_141, [432], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_875 = primals_141 = None
    getitem_265: "f32[8, 36, 1, 1]" = convolution_backward_47[0]
    getitem_266: "f32[432, 36, 1, 1]" = convolution_backward_47[1]
    getitem_267: "f32[432]" = convolution_backward_47[2];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_19: "b8[8, 36, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_19: "f32[8, 36, 1, 1]" = torch.ops.aten.where.self(le_19, full_default_5, getitem_265);  le_19 = getitem_265 = None
    unsqueeze_704: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
    unsqueeze_705: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_88: "f32[36]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_234: "f32[8, 36, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_706);  convolution_26 = unsqueeze_706 = None
    mul_876: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(where_19, sub_234)
    sum_89: "f32[36]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 2, 3]);  mul_876 = None
    mul_877: "f32[36]" = torch.ops.aten.mul.Tensor(sum_88, 0.125)
    unsqueeze_707: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
    unsqueeze_708: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_878: "f32[36]" = torch.ops.aten.mul.Tensor(sum_89, 0.125)
    mul_879: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_880: "f32[36]" = torch.ops.aten.mul.Tensor(mul_878, mul_879);  mul_878 = mul_879 = None
    unsqueeze_710: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_711: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_881: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_139);  primals_139 = None
    unsqueeze_713: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_714: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_882: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_712);  sub_234 = unsqueeze_712 = None
    sub_236: "f32[8, 36, 1, 1]" = torch.ops.aten.sub.Tensor(where_19, mul_882);  where_19 = mul_882 = None
    sub_237: "f32[8, 36, 1, 1]" = torch.ops.aten.sub.Tensor(sub_236, unsqueeze_709);  sub_236 = unsqueeze_709 = None
    mul_883: "f32[8, 36, 1, 1]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_715);  sub_237 = unsqueeze_715 = None
    mul_884: "f32[36]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_70);  sum_89 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_883, mean_3, primals_137, [36], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_883 = mean_3 = primals_137 = None
    getitem_268: "f32[8, 432, 1, 1]" = convolution_backward_48[0]
    getitem_269: "f32[36, 432, 1, 1]" = convolution_backward_48[1]
    getitem_270: "f32[36]" = convolution_backward_48[2];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 432, 14, 14]" = torch.ops.aten.expand.default(getitem_268, [8, 432, 14, 14]);  getitem_268 = None
    div_10: "f32[8, 432, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_357: "f32[8, 432, 14, 14]" = torch.ops.aten.add.Tensor(mul_873, div_10);  mul_873 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[432]" = torch.ops.aten.sum.dim_IntList(add_357, [0, 2, 3])
    sub_238: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_718);  convolution_25 = unsqueeze_718 = None
    mul_885: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(add_357, sub_238)
    sum_91: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 2, 3]);  mul_885 = None
    mul_886: "f32[432]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_886, 0);  mul_886 = None
    unsqueeze_720: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_887: "f32[432]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_888: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_889: "f32[432]" = torch.ops.aten.mul.Tensor(mul_887, mul_888);  mul_887 = mul_888 = None
    unsqueeze_722: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_723: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_890: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_39);  primals_39 = None
    unsqueeze_725: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_726: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_891: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_238, unsqueeze_724);  sub_238 = unsqueeze_724 = None
    sub_240: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(add_357, mul_891);  add_357 = mul_891 = None
    sub_241: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(sub_240, unsqueeze_721);  sub_240 = unsqueeze_721 = None
    mul_892: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_727);  sub_241 = unsqueeze_727 = None
    mul_893: "f32[432]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_67);  sum_91 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_892, mul_163, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 432, [True, True, False]);  mul_892 = mul_163 = primals_136 = None
    getitem_271: "f32[8, 432, 14, 14]" = convolution_backward_49[0]
    getitem_272: "f32[432, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_896: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_271, mul_895);  getitem_271 = mul_895 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 2, 3])
    sub_243: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_730);  convolution_24 = unsqueeze_730 = None
    mul_897: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(mul_896, sub_243)
    sum_93: "f32[432]" = torch.ops.aten.sum.dim_IntList(mul_897, [0, 2, 3]);  mul_897 = None
    mul_898: "f32[432]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_731: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_732: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_899: "f32[432]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_900: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_901: "f32[432]" = torch.ops.aten.mul.Tensor(mul_899, mul_900);  mul_899 = mul_900 = None
    unsqueeze_734: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_735: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_902: "f32[432]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_37);  primals_37 = None
    unsqueeze_737: "f32[1, 432]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_738: "f32[1, 432, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 432, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_903: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_736);  sub_243 = unsqueeze_736 = None
    sub_245: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(mul_896, mul_903);  mul_896 = mul_903 = None
    sub_246: "f32[8, 432, 14, 14]" = torch.ops.aten.sub.Tensor(sub_245, unsqueeze_733);  sub_245 = unsqueeze_733 = None
    mul_904: "f32[8, 432, 14, 14]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_739);  sub_246 = unsqueeze_739 = None
    mul_905: "f32[432]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_64);  sum_93 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_904, add_106, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = add_106 = primals_135 = None
    getitem_274: "f32[8, 72, 14, 14]" = convolution_backward_50[0]
    getitem_275: "f32[432, 72, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_359: "f32[8, 72, 14, 14]" = torch.ops.aten.add.Tensor(slice_61, getitem_274);  slice_61 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[72]" = torch.ops.aten.sum.dim_IntList(add_359, [0, 2, 3])
    sub_247: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_742);  convolution_23 = unsqueeze_742 = None
    mul_906: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(add_359, sub_247)
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_906, [0, 2, 3]);  mul_906 = None
    mul_907: "f32[72]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_743: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_744: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_908: "f32[72]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_909: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_910: "f32[72]" = torch.ops.aten.mul.Tensor(mul_908, mul_909);  mul_908 = mul_909 = None
    unsqueeze_746: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_747: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_911: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_35);  primals_35 = None
    unsqueeze_749: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_750: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_912: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_748);  sub_247 = unsqueeze_748 = None
    sub_249: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(add_359, mul_912);  add_359 = mul_912 = None
    sub_250: "f32[8, 72, 14, 14]" = torch.ops.aten.sub.Tensor(sub_249, unsqueeze_745);  sub_249 = unsqueeze_745 = None
    mul_913: "f32[8, 72, 14, 14]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_751);  sub_250 = unsqueeze_751 = None
    mul_914: "f32[72]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_61);  sum_95 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_913, clamp_max_5, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_913 = clamp_max_5 = primals_134 = None
    getitem_277: "f32[8, 366, 14, 14]" = convolution_backward_51[0]
    getitem_278: "f32[72, 366, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_20: "b8[8, 366, 14, 14]" = torch.ops.aten.le.Scalar(mul_148, 0.0)
    ge_10: "b8[8, 366, 14, 14]" = torch.ops.aten.ge.Scalar(mul_148, 6.0);  mul_148 = None
    bitwise_or_10: "b8[8, 366, 14, 14]" = torch.ops.aten.bitwise_or.Tensor(le_20, ge_10);  le_20 = ge_10 = None
    where_20: "f32[8, 366, 14, 14]" = torch.ops.aten.where.self(bitwise_or_10, full_default_5, getitem_277);  bitwise_or_10 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_915: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, add_96);  add_96 = None
    mul_916: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sigmoid_8);  where_20 = None
    sum_96: "f32[8, 366, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_915, [2, 3], True);  mul_915 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_251: "f32[8, 366, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
    mul_917: "f32[8, 366, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_251);  sigmoid_8 = sub_251 = None
    mul_918: "f32[8, 366, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_917);  sum_96 = mul_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_918, relu_2, primals_132, [366], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_918 = primals_132 = None
    getitem_280: "f32[8, 30, 1, 1]" = convolution_backward_52[0]
    getitem_281: "f32[366, 30, 1, 1]" = convolution_backward_52[1]
    getitem_282: "f32[366]" = convolution_backward_52[2];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_21: "b8[8, 30, 1, 1]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_21: "f32[8, 30, 1, 1]" = torch.ops.aten.where.self(le_21, full_default_5, getitem_280);  le_21 = getitem_280 = None
    unsqueeze_752: "f32[1, 30]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
    unsqueeze_753: "f32[1, 30, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_97: "f32[30]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_252: "f32[8, 30, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_754);  convolution_21 = unsqueeze_754 = None
    mul_919: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(where_21, sub_252)
    sum_98: "f32[30]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 2, 3]);  mul_919 = None
    mul_920: "f32[30]" = torch.ops.aten.mul.Tensor(sum_97, 0.125)
    unsqueeze_755: "f32[1, 30]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_756: "f32[1, 30, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_921: "f32[30]" = torch.ops.aten.mul.Tensor(sum_98, 0.125)
    mul_922: "f32[30]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_923: "f32[30]" = torch.ops.aten.mul.Tensor(mul_921, mul_922);  mul_921 = mul_922 = None
    unsqueeze_758: "f32[1, 30]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
    unsqueeze_759: "f32[1, 30, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_924: "f32[30]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_130);  primals_130 = None
    unsqueeze_761: "f32[1, 30]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_762: "f32[1, 30, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 30, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_925: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_760);  sub_252 = unsqueeze_760 = None
    sub_254: "f32[8, 30, 1, 1]" = torch.ops.aten.sub.Tensor(where_21, mul_925);  where_21 = mul_925 = None
    sub_255: "f32[8, 30, 1, 1]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_757);  sub_254 = unsqueeze_757 = None
    mul_926: "f32[8, 30, 1, 1]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_763);  sub_255 = unsqueeze_763 = None
    mul_927: "f32[30]" = torch.ops.aten.mul.Tensor(sum_98, squeeze_58);  sum_98 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_926, mean_2, primals_128, [30], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_926 = mean_2 = primals_128 = None
    getitem_283: "f32[8, 366, 1, 1]" = convolution_backward_53[0]
    getitem_284: "f32[30, 366, 1, 1]" = convolution_backward_53[1]
    getitem_285: "f32[30]" = convolution_backward_53[2];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 366, 14, 14]" = torch.ops.aten.expand.default(getitem_283, [8, 366, 14, 14]);  getitem_283 = None
    div_11: "f32[8, 366, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_360: "f32[8, 366, 14, 14]" = torch.ops.aten.add.Tensor(mul_916, div_11);  mul_916 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_99: "f32[366]" = torch.ops.aten.sum.dim_IntList(add_360, [0, 2, 3])
    sub_256: "f32[8, 366, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_766);  convolution_20 = unsqueeze_766 = None
    mul_928: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(add_360, sub_256)
    sum_100: "f32[366]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 2, 3]);  mul_928 = None
    mul_929: "f32[366]" = torch.ops.aten.mul.Tensor(sum_99, 0.0006377551020408163)
    unsqueeze_767: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_768: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_930: "f32[366]" = torch.ops.aten.mul.Tensor(sum_100, 0.0006377551020408163)
    mul_931: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_932: "f32[366]" = torch.ops.aten.mul.Tensor(mul_930, mul_931);  mul_930 = mul_931 = None
    unsqueeze_770: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
    unsqueeze_771: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_933: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_33);  primals_33 = None
    unsqueeze_773: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_774: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_934: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_772);  sub_256 = unsqueeze_772 = None
    sub_258: "f32[8, 366, 14, 14]" = torch.ops.aten.sub.Tensor(add_360, mul_934);  add_360 = mul_934 = None
    sub_259: "f32[8, 366, 14, 14]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_769);  sub_258 = unsqueeze_769 = None
    mul_935: "f32[8, 366, 14, 14]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_775);  sub_259 = unsqueeze_775 = None
    mul_936: "f32[366]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_55);  sum_100 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_935, mul_133, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 366, [True, True, False]);  mul_935 = mul_133 = primals_127 = None
    getitem_286: "f32[8, 366, 28, 28]" = convolution_backward_54[0]
    getitem_287: "f32[366, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_939: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_286, mul_938);  getitem_286 = mul_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_101: "f32[366]" = torch.ops.aten.sum.dim_IntList(mul_939, [0, 2, 3])
    sub_261: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_778);  convolution_19 = unsqueeze_778 = None
    mul_940: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(mul_939, sub_261)
    sum_102: "f32[366]" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
    mul_941: "f32[366]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    unsqueeze_779: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_780: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_942: "f32[366]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    mul_943: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_944: "f32[366]" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    unsqueeze_782: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_783: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_945: "f32[366]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_31);  primals_31 = None
    unsqueeze_785: "f32[1, 366]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_786: "f32[1, 366, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 366, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_946: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_784);  sub_261 = unsqueeze_784 = None
    sub_263: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(mul_939, mul_946);  mul_939 = mul_946 = None
    sub_264: "f32[8, 366, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_781);  sub_263 = unsqueeze_781 = None
    mul_947: "f32[8, 366, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_787);  sub_264 = unsqueeze_787 = None
    mul_948: "f32[366]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_52);  sum_102 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_947, cat_1, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = cat_1 = primals_126 = None
    getitem_289: "f32[8, 61, 28, 28]" = convolution_backward_55[0]
    getitem_290: "f32[366, 61, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_63: "f32[8, 50, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_289, 1, 0, 50)
    slice_64: "f32[8, 11, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_289, 1, 50, 61);  getitem_289 = None
    full_default_70: "f32[8, 61, 28, 28]" = torch.ops.aten.full.default([8, 61, 28, 28], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_36: "f32[8, 61, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_70, slice_64, 1, 50, 9223372036854775807);  slice_64 = None
    slice_scatter_38: "f32[8, 61, 28, 28]" = torch.ops.aten.slice_scatter.default(full_default_70, slice_63, 1, 0, 50);  full_default_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_362: "f32[8, 61, 28, 28]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_38);  slice_scatter_36 = slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_103: "f32[61]" = torch.ops.aten.sum.dim_IntList(add_362, [0, 2, 3])
    sub_265: "f32[8, 61, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_790);  convolution_18 = unsqueeze_790 = None
    mul_949: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(add_362, sub_265)
    sum_104: "f32[61]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
    mul_950: "f32[61]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    unsqueeze_791: "f32[1, 61]" = torch.ops.aten.unsqueeze.default(mul_950, 0);  mul_950 = None
    unsqueeze_792: "f32[1, 61, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_951: "f32[61]" = torch.ops.aten.mul.Tensor(sum_104, 0.00015943877551020407)
    mul_952: "f32[61]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_953: "f32[61]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_794: "f32[1, 61]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_795: "f32[1, 61, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_954: "f32[61]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_29);  primals_29 = None
    unsqueeze_797: "f32[1, 61]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_798: "f32[1, 61, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 61, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_955: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_796);  sub_265 = unsqueeze_796 = None
    sub_267: "f32[8, 61, 28, 28]" = torch.ops.aten.sub.Tensor(add_362, mul_955);  add_362 = mul_955 = None
    sub_268: "f32[8, 61, 28, 28]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_793);  sub_267 = unsqueeze_793 = None
    mul_956: "f32[8, 61, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_799);  sub_268 = unsqueeze_799 = None
    mul_957: "f32[61]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_49);  sum_104 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_956, clamp_max_4, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_956 = clamp_max_4 = primals_125 = None
    getitem_292: "f32[8, 300, 28, 28]" = convolution_backward_56[0]
    getitem_293: "f32[61, 300, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_22: "b8[8, 300, 28, 28]" = torch.ops.aten.le.Scalar(mul_118, 0.0)
    ge_11: "b8[8, 300, 28, 28]" = torch.ops.aten.ge.Scalar(mul_118, 6.0);  mul_118 = None
    bitwise_or_11: "b8[8, 300, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_22, ge_11);  le_22 = ge_11 = None
    where_22: "f32[8, 300, 28, 28]" = torch.ops.aten.where.self(bitwise_or_11, full_default_5, getitem_292);  bitwise_or_11 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_958: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, add_75);  add_75 = None
    mul_959: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sigmoid_6);  where_22 = None
    sum_105: "f32[8, 300, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_958, [2, 3], True);  mul_958 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_269: "f32[8, 300, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
    mul_960: "f32[8, 300, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_269);  sigmoid_6 = sub_269 = None
    mul_961: "f32[8, 300, 1, 1]" = torch.ops.aten.mul.Tensor(sum_105, mul_960);  sum_105 = mul_960 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_961, relu_1, primals_123, [300], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_961 = primals_123 = None
    getitem_295: "f32[8, 25, 1, 1]" = convolution_backward_57[0]
    getitem_296: "f32[300, 25, 1, 1]" = convolution_backward_57[1]
    getitem_297: "f32[300]" = convolution_backward_57[2];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_23: "b8[8, 25, 1, 1]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_23: "f32[8, 25, 1, 1]" = torch.ops.aten.where.self(le_23, full_default_5, getitem_295);  le_23 = getitem_295 = None
    unsqueeze_800: "f32[1, 25]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
    unsqueeze_801: "f32[1, 25, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_106: "f32[25]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_270: "f32[8, 25, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_802);  convolution_16 = unsqueeze_802 = None
    mul_962: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(where_23, sub_270)
    sum_107: "f32[25]" = torch.ops.aten.sum.dim_IntList(mul_962, [0, 2, 3]);  mul_962 = None
    mul_963: "f32[25]" = torch.ops.aten.mul.Tensor(sum_106, 0.125)
    unsqueeze_803: "f32[1, 25]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_804: "f32[1, 25, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_964: "f32[25]" = torch.ops.aten.mul.Tensor(sum_107, 0.125)
    mul_965: "f32[25]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_966: "f32[25]" = torch.ops.aten.mul.Tensor(mul_964, mul_965);  mul_964 = mul_965 = None
    unsqueeze_806: "f32[1, 25]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_807: "f32[1, 25, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_967: "f32[25]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_121);  primals_121 = None
    unsqueeze_809: "f32[1, 25]" = torch.ops.aten.unsqueeze.default(mul_967, 0);  mul_967 = None
    unsqueeze_810: "f32[1, 25, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 25, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_968: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(sub_270, unsqueeze_808);  sub_270 = unsqueeze_808 = None
    sub_272: "f32[8, 25, 1, 1]" = torch.ops.aten.sub.Tensor(where_23, mul_968);  where_23 = mul_968 = None
    sub_273: "f32[8, 25, 1, 1]" = torch.ops.aten.sub.Tensor(sub_272, unsqueeze_805);  sub_272 = unsqueeze_805 = None
    mul_969: "f32[8, 25, 1, 1]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_811);  sub_273 = unsqueeze_811 = None
    mul_970: "f32[25]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_46);  sum_107 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_969, mean_1, primals_119, [25], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_969 = mean_1 = primals_119 = None
    getitem_298: "f32[8, 300, 1, 1]" = convolution_backward_58[0]
    getitem_299: "f32[25, 300, 1, 1]" = convolution_backward_58[1]
    getitem_300: "f32[25]" = convolution_backward_58[2];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 300, 28, 28]" = torch.ops.aten.expand.default(getitem_298, [8, 300, 28, 28]);  getitem_298 = None
    div_12: "f32[8, 300, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_363: "f32[8, 300, 28, 28]" = torch.ops.aten.add.Tensor(mul_959, div_12);  mul_959 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[300]" = torch.ops.aten.sum.dim_IntList(add_363, [0, 2, 3])
    sub_274: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_814);  convolution_15 = unsqueeze_814 = None
    mul_971: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(add_363, sub_274)
    sum_109: "f32[300]" = torch.ops.aten.sum.dim_IntList(mul_971, [0, 2, 3]);  mul_971 = None
    mul_972: "f32[300]" = torch.ops.aten.mul.Tensor(sum_108, 0.00015943877551020407)
    unsqueeze_815: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_816: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_973: "f32[300]" = torch.ops.aten.mul.Tensor(sum_109, 0.00015943877551020407)
    mul_974: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_975: "f32[300]" = torch.ops.aten.mul.Tensor(mul_973, mul_974);  mul_973 = mul_974 = None
    unsqueeze_818: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_819: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_976: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_27);  primals_27 = None
    unsqueeze_821: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_976, 0);  mul_976 = None
    unsqueeze_822: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_977: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_820);  sub_274 = unsqueeze_820 = None
    sub_276: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(add_363, mul_977);  add_363 = mul_977 = None
    sub_277: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_817);  sub_276 = unsqueeze_817 = None
    mul_978: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_823);  sub_277 = unsqueeze_823 = None
    mul_979: "f32[300]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_43);  sum_109 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_978, mul_103, primals_118, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 300, [True, True, False]);  mul_978 = mul_103 = primals_118 = None
    getitem_301: "f32[8, 300, 28, 28]" = convolution_backward_59[0]
    getitem_302: "f32[300, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_982: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_301, mul_981);  getitem_301 = mul_981 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[300]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 2, 3])
    sub_279: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_826);  convolution_14 = unsqueeze_826 = None
    mul_983: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(mul_982, sub_279)
    sum_111: "f32[300]" = torch.ops.aten.sum.dim_IntList(mul_983, [0, 2, 3]);  mul_983 = None
    mul_984: "f32[300]" = torch.ops.aten.mul.Tensor(sum_110, 0.00015943877551020407)
    unsqueeze_827: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_828: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_985: "f32[300]" = torch.ops.aten.mul.Tensor(sum_111, 0.00015943877551020407)
    mul_986: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_987: "f32[300]" = torch.ops.aten.mul.Tensor(mul_985, mul_986);  mul_985 = mul_986 = None
    unsqueeze_830: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_831: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_988: "f32[300]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_25);  primals_25 = None
    unsqueeze_833: "f32[1, 300]" = torch.ops.aten.unsqueeze.default(mul_988, 0);  mul_988 = None
    unsqueeze_834: "f32[1, 300, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 300, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_989: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_832);  sub_279 = unsqueeze_832 = None
    sub_281: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(mul_982, mul_989);  mul_982 = mul_989 = None
    sub_282: "f32[8, 300, 28, 28]" = torch.ops.aten.sub.Tensor(sub_281, unsqueeze_829);  sub_281 = unsqueeze_829 = None
    mul_990: "f32[8, 300, 28, 28]" = torch.ops.aten.mul.Tensor(sub_282, unsqueeze_835);  sub_282 = unsqueeze_835 = None
    mul_991: "f32[300]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_40);  sum_111 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_990, add_65, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_990 = add_65 = primals_117 = None
    getitem_304: "f32[8, 50, 28, 28]" = convolution_backward_60[0]
    getitem_305: "f32[300, 50, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_365: "f32[8, 50, 28, 28]" = torch.ops.aten.add.Tensor(slice_63, getitem_304);  slice_63 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[50]" = torch.ops.aten.sum.dim_IntList(add_365, [0, 2, 3])
    sub_283: "f32[8, 50, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_838);  convolution_13 = unsqueeze_838 = None
    mul_992: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(add_365, sub_283)
    sum_113: "f32[50]" = torch.ops.aten.sum.dim_IntList(mul_992, [0, 2, 3]);  mul_992 = None
    mul_993: "f32[50]" = torch.ops.aten.mul.Tensor(sum_112, 0.00015943877551020407)
    unsqueeze_839: "f32[1, 50]" = torch.ops.aten.unsqueeze.default(mul_993, 0);  mul_993 = None
    unsqueeze_840: "f32[1, 50, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_994: "f32[50]" = torch.ops.aten.mul.Tensor(sum_113, 0.00015943877551020407)
    mul_995: "f32[50]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_996: "f32[50]" = torch.ops.aten.mul.Tensor(mul_994, mul_995);  mul_994 = mul_995 = None
    unsqueeze_842: "f32[1, 50]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_843: "f32[1, 50, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_997: "f32[50]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_23);  primals_23 = None
    unsqueeze_845: "f32[1, 50]" = torch.ops.aten.unsqueeze.default(mul_997, 0);  mul_997 = None
    unsqueeze_846: "f32[1, 50, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 50, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_998: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_844);  sub_283 = unsqueeze_844 = None
    sub_285: "f32[8, 50, 28, 28]" = torch.ops.aten.sub.Tensor(add_365, mul_998);  add_365 = mul_998 = None
    sub_286: "f32[8, 50, 28, 28]" = torch.ops.aten.sub.Tensor(sub_285, unsqueeze_841);  sub_285 = unsqueeze_841 = None
    mul_999: "f32[8, 50, 28, 28]" = torch.ops.aten.mul.Tensor(sub_286, unsqueeze_847);  sub_286 = unsqueeze_847 = None
    mul_1000: "f32[50]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_37);  sum_113 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_999, clamp_max_3, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_999 = clamp_max_3 = primals_116 = None
    getitem_307: "f32[8, 228, 28, 28]" = convolution_backward_61[0]
    getitem_308: "f32[50, 228, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    le_24: "b8[8, 228, 28, 28]" = torch.ops.aten.le.Scalar(mul_88, 0.0)
    ge_12: "b8[8, 228, 28, 28]" = torch.ops.aten.ge.Scalar(mul_88, 6.0);  mul_88 = None
    bitwise_or_12: "b8[8, 228, 28, 28]" = torch.ops.aten.bitwise_or.Tensor(le_24, ge_12);  le_24 = ge_12 = None
    where_24: "f32[8, 228, 28, 28]" = torch.ops.aten.where.self(bitwise_or_12, full_default_5, getitem_307);  bitwise_or_12 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_1001: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, add_55);  add_55 = None
    mul_1002: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sigmoid_4);  where_24 = None
    sum_114: "f32[8, 228, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [2, 3], True);  mul_1001 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_287: "f32[8, 228, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
    mul_1003: "f32[8, 228, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_287);  sigmoid_4 = sub_287 = None
    mul_1004: "f32[8, 228, 1, 1]" = torch.ops.aten.mul.Tensor(sum_114, mul_1003);  sum_114 = mul_1003 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1004, relu, primals_114, [228], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1004 = primals_114 = None
    getitem_310: "f32[8, 19, 1, 1]" = convolution_backward_62[0]
    getitem_311: "f32[228, 19, 1, 1]" = convolution_backward_62[1]
    getitem_312: "f32[228]" = convolution_backward_62[2];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_25: "b8[8, 19, 1, 1]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_25: "f32[8, 19, 1, 1]" = torch.ops.aten.where.self(le_25, full_default_5, getitem_310);  le_25 = getitem_310 = None
    unsqueeze_848: "f32[1, 19]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
    unsqueeze_849: "f32[1, 19, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_115: "f32[19]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_288: "f32[8, 19, 1, 1]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_850);  convolution_11 = unsqueeze_850 = None
    mul_1005: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(where_25, sub_288)
    sum_116: "f32[19]" = torch.ops.aten.sum.dim_IntList(mul_1005, [0, 2, 3]);  mul_1005 = None
    mul_1006: "f32[19]" = torch.ops.aten.mul.Tensor(sum_115, 0.125)
    unsqueeze_851: "f32[1, 19]" = torch.ops.aten.unsqueeze.default(mul_1006, 0);  mul_1006 = None
    unsqueeze_852: "f32[1, 19, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_1007: "f32[19]" = torch.ops.aten.mul.Tensor(sum_116, 0.125)
    mul_1008: "f32[19]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_1009: "f32[19]" = torch.ops.aten.mul.Tensor(mul_1007, mul_1008);  mul_1007 = mul_1008 = None
    unsqueeze_854: "f32[1, 19]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_855: "f32[1, 19, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_1010: "f32[19]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_112);  primals_112 = None
    unsqueeze_857: "f32[1, 19]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_858: "f32[1, 19, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 19, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_1011: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_856);  sub_288 = unsqueeze_856 = None
    sub_290: "f32[8, 19, 1, 1]" = torch.ops.aten.sub.Tensor(where_25, mul_1011);  where_25 = mul_1011 = None
    sub_291: "f32[8, 19, 1, 1]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_853);  sub_290 = unsqueeze_853 = None
    mul_1012: "f32[8, 19, 1, 1]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_859);  sub_291 = unsqueeze_859 = None
    mul_1013: "f32[19]" = torch.ops.aten.mul.Tensor(sum_116, squeeze_34);  sum_116 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1012, mean, primals_110, [19], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_1012 = mean = primals_110 = None
    getitem_313: "f32[8, 228, 1, 1]" = convolution_backward_63[0]
    getitem_314: "f32[19, 228, 1, 1]" = convolution_backward_63[1]
    getitem_315: "f32[19]" = convolution_backward_63[2];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 228, 28, 28]" = torch.ops.aten.expand.default(getitem_313, [8, 228, 28, 28]);  getitem_313 = None
    div_13: "f32[8, 228, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_366: "f32[8, 228, 28, 28]" = torch.ops.aten.add.Tensor(mul_1002, div_13);  mul_1002 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_117: "f32[228]" = torch.ops.aten.sum.dim_IntList(add_366, [0, 2, 3])
    sub_292: "f32[8, 228, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_862);  convolution_10 = unsqueeze_862 = None
    mul_1014: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(add_366, sub_292)
    sum_118: "f32[228]" = torch.ops.aten.sum.dim_IntList(mul_1014, [0, 2, 3]);  mul_1014 = None
    mul_1015: "f32[228]" = torch.ops.aten.mul.Tensor(sum_117, 0.00015943877551020407)
    unsqueeze_863: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1015, 0);  mul_1015 = None
    unsqueeze_864: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_1016: "f32[228]" = torch.ops.aten.mul.Tensor(sum_118, 0.00015943877551020407)
    mul_1017: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_1018: "f32[228]" = torch.ops.aten.mul.Tensor(mul_1016, mul_1017);  mul_1016 = mul_1017 = None
    unsqueeze_866: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_867: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_1019: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_869: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_870: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_1020: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_868);  sub_292 = unsqueeze_868 = None
    sub_294: "f32[8, 228, 28, 28]" = torch.ops.aten.sub.Tensor(add_366, mul_1020);  add_366 = mul_1020 = None
    sub_295: "f32[8, 228, 28, 28]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_865);  sub_294 = unsqueeze_865 = None
    mul_1021: "f32[8, 228, 28, 28]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_871);  sub_295 = unsqueeze_871 = None
    mul_1022: "f32[228]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_31);  sum_118 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1021, mul_73, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 228, [True, True, False]);  mul_1021 = mul_73 = primals_109 = None
    getitem_316: "f32[8, 228, 56, 56]" = convolution_backward_64[0]
    getitem_317: "f32[228, 1, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1025: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_316, mul_1024);  getitem_316 = mul_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_119: "f32[228]" = torch.ops.aten.sum.dim_IntList(mul_1025, [0, 2, 3])
    sub_297: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_874);  convolution_9 = unsqueeze_874 = None
    mul_1026: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1025, sub_297)
    sum_120: "f32[228]" = torch.ops.aten.sum.dim_IntList(mul_1026, [0, 2, 3]);  mul_1026 = None
    mul_1027: "f32[228]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    unsqueeze_875: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_876: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_1028: "f32[228]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
    mul_1029: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_1030: "f32[228]" = torch.ops.aten.mul.Tensor(mul_1028, mul_1029);  mul_1028 = mul_1029 = None
    unsqueeze_878: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1030, 0);  mul_1030 = None
    unsqueeze_879: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_1031: "f32[228]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_881: "f32[1, 228]" = torch.ops.aten.unsqueeze.default(mul_1031, 0);  mul_1031 = None
    unsqueeze_882: "f32[1, 228, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 228, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_1032: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_880);  sub_297 = unsqueeze_880 = None
    sub_299: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(mul_1025, mul_1032);  mul_1025 = mul_1032 = None
    sub_300: "f32[8, 228, 56, 56]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_877);  sub_299 = unsqueeze_877 = None
    mul_1033: "f32[8, 228, 56, 56]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_883);  sub_300 = unsqueeze_883 = None
    mul_1034: "f32[228]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_28);  sum_120 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_1033, cat, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1033 = cat = primals_108 = None
    getitem_319: "f32[8, 38, 56, 56]" = convolution_backward_65[0]
    getitem_320: "f32[228, 38, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    slice_65: "f32[8, 27, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_319, 1, 0, 27)
    slice_66: "f32[8, 11, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_319, 1, 27, 38);  getitem_319 = None
    full_default_80: "f32[8, 38, 56, 56]" = torch.ops.aten.full.default([8, 38, 56, 56], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_40: "f32[8, 38, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_80, slice_66, 1, 27, 9223372036854775807);  slice_66 = None
    slice_scatter_42: "f32[8, 38, 56, 56]" = torch.ops.aten.slice_scatter.default(full_default_80, slice_65, 1, 0, 27);  full_default_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:91, code: x = torch.cat([x[:, 0:self.in_channels] + shortcut, x[:, self.in_channels:]], dim=1)
    add_368: "f32[8, 38, 56, 56]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_42);  slice_scatter_40 = slice_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_121: "f32[38]" = torch.ops.aten.sum.dim_IntList(add_368, [0, 2, 3])
    sub_301: "f32[8, 38, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_886);  convolution_8 = unsqueeze_886 = None
    mul_1035: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(add_368, sub_301)
    sum_122: "f32[38]" = torch.ops.aten.sum.dim_IntList(mul_1035, [0, 2, 3]);  mul_1035 = None
    mul_1036: "f32[38]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
    unsqueeze_887: "f32[1, 38]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_888: "f32[1, 38, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_1037: "f32[38]" = torch.ops.aten.mul.Tensor(sum_122, 3.985969387755102e-05)
    mul_1038: "f32[38]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_1039: "f32[38]" = torch.ops.aten.mul.Tensor(mul_1037, mul_1038);  mul_1037 = mul_1038 = None
    unsqueeze_890: "f32[1, 38]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
    unsqueeze_891: "f32[1, 38, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_1040: "f32[38]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_893: "f32[1, 38]" = torch.ops.aten.unsqueeze.default(mul_1040, 0);  mul_1040 = None
    unsqueeze_894: "f32[1, 38, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 38, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_1041: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_892);  sub_301 = unsqueeze_892 = None
    sub_303: "f32[8, 38, 56, 56]" = torch.ops.aten.sub.Tensor(add_368, mul_1041);  add_368 = mul_1041 = None
    sub_304: "f32[8, 38, 56, 56]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_889);  sub_303 = unsqueeze_889 = None
    mul_1042: "f32[8, 38, 56, 56]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_895);  sub_304 = unsqueeze_895 = None
    mul_1043: "f32[38]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_25);  sum_122 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_1042, clamp_max_2, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1042 = clamp_max_2 = primals_107 = None
    getitem_322: "f32[8, 162, 56, 56]" = convolution_backward_66[0]
    getitem_323: "f32[38, 162, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    where_26: "f32[8, 162, 56, 56]" = torch.ops.aten.where.self(bitwise_or_13, full_default_5, getitem_322);  bitwise_or_13 = getitem_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_123: "f32[162]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_305: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_898);  convolution_7 = unsqueeze_898 = None
    mul_1044: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, sub_305)
    sum_124: "f32[162]" = torch.ops.aten.sum.dim_IntList(mul_1044, [0, 2, 3]);  mul_1044 = None
    mul_1045: "f32[162]" = torch.ops.aten.mul.Tensor(sum_123, 3.985969387755102e-05)
    unsqueeze_899: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1045, 0);  mul_1045 = None
    unsqueeze_900: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_1046: "f32[162]" = torch.ops.aten.mul.Tensor(sum_124, 3.985969387755102e-05)
    mul_1047: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_1048: "f32[162]" = torch.ops.aten.mul.Tensor(mul_1046, mul_1047);  mul_1046 = mul_1047 = None
    unsqueeze_902: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1048, 0);  mul_1048 = None
    unsqueeze_903: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_1049: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_905: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1049, 0);  mul_1049 = None
    unsqueeze_906: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_1050: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_904);  sub_305 = unsqueeze_904 = None
    sub_307: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(where_26, mul_1050);  where_26 = mul_1050 = None
    sub_308: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_901);  sub_307 = unsqueeze_901 = None
    mul_1051: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_907);  sub_308 = unsqueeze_907 = None
    mul_1052: "f32[162]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_22);  sum_124 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_1051, mul_51, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 162, [True, True, False]);  mul_1051 = mul_51 = primals_106 = None
    getitem_325: "f32[8, 162, 56, 56]" = convolution_backward_67[0]
    getitem_326: "f32[162, 1, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1055: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_325, mul_1054);  getitem_325 = mul_1054 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_125: "f32[162]" = torch.ops.aten.sum.dim_IntList(mul_1055, [0, 2, 3])
    sub_310: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_910);  convolution_6 = unsqueeze_910 = None
    mul_1056: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(mul_1055, sub_310)
    sum_126: "f32[162]" = torch.ops.aten.sum.dim_IntList(mul_1056, [0, 2, 3]);  mul_1056 = None
    mul_1057: "f32[162]" = torch.ops.aten.mul.Tensor(sum_125, 3.985969387755102e-05)
    unsqueeze_911: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1057, 0);  mul_1057 = None
    unsqueeze_912: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_1058: "f32[162]" = torch.ops.aten.mul.Tensor(sum_126, 3.985969387755102e-05)
    mul_1059: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_1060: "f32[162]" = torch.ops.aten.mul.Tensor(mul_1058, mul_1059);  mul_1058 = mul_1059 = None
    unsqueeze_914: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1060, 0);  mul_1060 = None
    unsqueeze_915: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_1061: "f32[162]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_917: "f32[1, 162]" = torch.ops.aten.unsqueeze.default(mul_1061, 0);  mul_1061 = None
    unsqueeze_918: "f32[1, 162, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 162, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_1062: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_310, unsqueeze_916);  sub_310 = unsqueeze_916 = None
    sub_312: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(mul_1055, mul_1062);  mul_1055 = mul_1062 = None
    sub_313: "f32[8, 162, 56, 56]" = torch.ops.aten.sub.Tensor(sub_312, unsqueeze_913);  sub_312 = unsqueeze_913 = None
    mul_1063: "f32[8, 162, 56, 56]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_919);  sub_313 = unsqueeze_919 = None
    mul_1064: "f32[162]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_19);  sum_126 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_1063, add_29, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1063 = add_29 = primals_105 = None
    getitem_328: "f32[8, 27, 56, 56]" = convolution_backward_68[0]
    getitem_329: "f32[162, 27, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_370: "f32[8, 27, 56, 56]" = torch.ops.aten.add.Tensor(slice_65, getitem_328);  slice_65 = getitem_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_127: "f32[27]" = torch.ops.aten.sum.dim_IntList(add_370, [0, 2, 3])
    sub_314: "f32[8, 27, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_922);  convolution_5 = unsqueeze_922 = None
    mul_1065: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(add_370, sub_314)
    sum_128: "f32[27]" = torch.ops.aten.sum.dim_IntList(mul_1065, [0, 2, 3]);  mul_1065 = None
    mul_1066: "f32[27]" = torch.ops.aten.mul.Tensor(sum_127, 3.985969387755102e-05)
    unsqueeze_923: "f32[1, 27]" = torch.ops.aten.unsqueeze.default(mul_1066, 0);  mul_1066 = None
    unsqueeze_924: "f32[1, 27, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_1067: "f32[27]" = torch.ops.aten.mul.Tensor(sum_128, 3.985969387755102e-05)
    mul_1068: "f32[27]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_1069: "f32[27]" = torch.ops.aten.mul.Tensor(mul_1067, mul_1068);  mul_1067 = mul_1068 = None
    unsqueeze_926: "f32[1, 27]" = torch.ops.aten.unsqueeze.default(mul_1069, 0);  mul_1069 = None
    unsqueeze_927: "f32[1, 27, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_1070: "f32[27]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_929: "f32[1, 27]" = torch.ops.aten.unsqueeze.default(mul_1070, 0);  mul_1070 = None
    unsqueeze_930: "f32[1, 27, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 27, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_1071: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(sub_314, unsqueeze_928);  sub_314 = unsqueeze_928 = None
    sub_316: "f32[8, 27, 56, 56]" = torch.ops.aten.sub.Tensor(add_370, mul_1071);  add_370 = mul_1071 = None
    sub_317: "f32[8, 27, 56, 56]" = torch.ops.aten.sub.Tensor(sub_316, unsqueeze_925);  sub_316 = unsqueeze_925 = None
    mul_1072: "f32[8, 27, 56, 56]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_931);  sub_317 = unsqueeze_931 = None
    mul_1073: "f32[27]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_16);  sum_128 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_1072, clamp_max_1, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1072 = clamp_max_1 = primals_104 = None
    getitem_331: "f32[8, 96, 56, 56]" = convolution_backward_69[0]
    getitem_332: "f32[27, 96, 1, 1]" = convolution_backward_69[1];  convolution_backward_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    where_27: "f32[8, 96, 56, 56]" = torch.ops.aten.where.self(bitwise_or_14, full_default_5, getitem_331);  bitwise_or_14 = getitem_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_129: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_318: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_934);  convolution_4 = unsqueeze_934 = None
    mul_1074: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, sub_318)
    sum_130: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1074, [0, 2, 3]);  mul_1074 = None
    mul_1075: "f32[96]" = torch.ops.aten.mul.Tensor(sum_129, 3.985969387755102e-05)
    unsqueeze_935: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1075, 0);  mul_1075 = None
    unsqueeze_936: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_1076: "f32[96]" = torch.ops.aten.mul.Tensor(sum_130, 3.985969387755102e-05)
    mul_1077: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_1078: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1076, mul_1077);  mul_1076 = mul_1077 = None
    unsqueeze_938: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1078, 0);  mul_1078 = None
    unsqueeze_939: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_1079: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_941: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1079, 0);  mul_1079 = None
    unsqueeze_942: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_1080: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_318, unsqueeze_940);  sub_318 = unsqueeze_940 = None
    sub_320: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_27, mul_1080);  where_27 = mul_1080 = None
    sub_321: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_320, unsqueeze_937);  sub_320 = unsqueeze_937 = None
    mul_1081: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_943);  sub_321 = unsqueeze_943 = None
    mul_1082: "f32[96]" = torch.ops.aten.mul.Tensor(sum_130, squeeze_13);  sum_130 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_1081, mul_29, primals_103, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_1081 = mul_29 = primals_103 = None
    getitem_334: "f32[8, 96, 112, 112]" = convolution_backward_70[0]
    getitem_335: "f32[96, 1, 3, 3]" = convolution_backward_70[1];  convolution_backward_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1085: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_334, mul_1084);  getitem_334 = mul_1084 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_131: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1085, [0, 2, 3])
    sub_323: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_946);  convolution_3 = unsqueeze_946 = None
    mul_1086: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1085, sub_323)
    sum_132: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1086, [0, 2, 3]);  mul_1086 = None
    mul_1087: "f32[96]" = torch.ops.aten.mul.Tensor(sum_131, 9.964923469387754e-06)
    unsqueeze_947: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1087, 0);  mul_1087 = None
    unsqueeze_948: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_1088: "f32[96]" = torch.ops.aten.mul.Tensor(sum_132, 9.964923469387754e-06)
    mul_1089: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1090: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1088, mul_1089);  mul_1088 = mul_1089 = None
    unsqueeze_950: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1090, 0);  mul_1090 = None
    unsqueeze_951: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_1091: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_953: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1091, 0);  mul_1091 = None
    unsqueeze_954: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_1092: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_323, unsqueeze_952);  sub_323 = unsqueeze_952 = None
    sub_325: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1085, mul_1092);  mul_1085 = mul_1092 = None
    sub_326: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(sub_325, unsqueeze_949);  sub_325 = unsqueeze_949 = None
    mul_1093: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_326, unsqueeze_955);  sub_326 = unsqueeze_955 = None
    mul_1094: "f32[96]" = torch.ops.aten.mul.Tensor(sum_132, squeeze_10);  sum_132 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_1093, add_14, primals_102, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1093 = add_14 = primals_102 = None
    getitem_337: "f32[8, 16, 112, 112]" = convolution_backward_71[0]
    getitem_338: "f32[96, 16, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_133: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_337, [0, 2, 3])
    sub_327: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_958);  convolution_2 = unsqueeze_958 = None
    mul_1095: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_337, sub_327)
    sum_134: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1095, [0, 2, 3]);  mul_1095 = None
    mul_1096: "f32[16]" = torch.ops.aten.mul.Tensor(sum_133, 9.964923469387754e-06)
    unsqueeze_959: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1096, 0);  mul_1096 = None
    unsqueeze_960: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_1097: "f32[16]" = torch.ops.aten.mul.Tensor(sum_134, 9.964923469387754e-06)
    mul_1098: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1099: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1097, mul_1098);  mul_1097 = mul_1098 = None
    unsqueeze_962: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1099, 0);  mul_1099 = None
    unsqueeze_963: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_1100: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_965: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1100, 0);  mul_1100 = None
    unsqueeze_966: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_1101: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_327, unsqueeze_964);  sub_327 = unsqueeze_964 = None
    sub_329: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_337, mul_1101);  getitem_337 = mul_1101 = None
    sub_330: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_329, unsqueeze_961);  sub_329 = unsqueeze_961 = None
    mul_1102: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_330, unsqueeze_967);  sub_330 = unsqueeze_967 = None
    mul_1103: "f32[16]" = torch.ops.aten.mul.Tensor(sum_134, squeeze_7);  sum_134 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_1102, clamp_max, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1102 = clamp_max = primals_101 = None
    getitem_340: "f32[8, 32, 112, 112]" = convolution_backward_72[0]
    getitem_341: "f32[16, 32, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/rexnet.py:86, code: x = self.act_dw(x)
    where_28: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(bitwise_or_15, full_default_5, getitem_340);  bitwise_or_15 = full_default_5 = getitem_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_135: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_331: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_970);  convolution_1 = unsqueeze_970 = None
    mul_1104: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_28, sub_331)
    sum_136: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1104, [0, 2, 3]);  mul_1104 = None
    mul_1105: "f32[32]" = torch.ops.aten.mul.Tensor(sum_135, 9.964923469387754e-06)
    unsqueeze_971: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1105, 0);  mul_1105 = None
    unsqueeze_972: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_1106: "f32[32]" = torch.ops.aten.mul.Tensor(sum_136, 9.964923469387754e-06)
    mul_1107: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1108: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1106, mul_1107);  mul_1106 = mul_1107 = None
    unsqueeze_974: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1108, 0);  mul_1108 = None
    unsqueeze_975: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_1109: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_977: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1109, 0);  mul_1109 = None
    unsqueeze_978: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_1110: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_331, unsqueeze_976);  sub_331 = unsqueeze_976 = None
    sub_333: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_28, mul_1110);  where_28 = mul_1110 = None
    sub_334: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_333, unsqueeze_973);  sub_333 = unsqueeze_973 = None
    mul_1111: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_334, unsqueeze_979);  sub_334 = unsqueeze_979 = None
    mul_1112: "f32[32]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_4);  sum_136 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_1111, mul_7, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1111 = mul_7 = primals_100 = None
    getitem_343: "f32[8, 32, 112, 112]" = convolution_backward_73[0]
    getitem_344: "f32[32, 1, 3, 3]" = convolution_backward_73[1];  convolution_backward_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_1115: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_343, mul_1114);  getitem_343 = mul_1114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_137: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1115, [0, 2, 3])
    sub_336: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_982);  convolution = unsqueeze_982 = None
    mul_1116: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1115, sub_336)
    sum_138: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1116, [0, 2, 3]);  mul_1116 = None
    mul_1117: "f32[32]" = torch.ops.aten.mul.Tensor(sum_137, 9.964923469387754e-06)
    unsqueeze_983: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1117, 0);  mul_1117 = None
    unsqueeze_984: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_1118: "f32[32]" = torch.ops.aten.mul.Tensor(sum_138, 9.964923469387754e-06)
    mul_1119: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1120: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1118, mul_1119);  mul_1118 = mul_1119 = None
    unsqueeze_986: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1120, 0);  mul_1120 = None
    unsqueeze_987: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1121: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_989: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1121, 0);  mul_1121 = None
    unsqueeze_990: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1122: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_336, unsqueeze_988);  sub_336 = unsqueeze_988 = None
    sub_338: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1115, mul_1122);  mul_1115 = mul_1122 = None
    sub_339: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_338, unsqueeze_985);  sub_338 = unsqueeze_985 = None
    mul_1123: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_339, unsqueeze_991);  sub_339 = unsqueeze_991 = None
    mul_1124: "f32[32]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_1);  sum_138 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1123, primals_414, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1123 = primals_414 = primals_99 = None
    getitem_347: "f32[32, 3, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None
    return [mul_1124, sum_137, mul_1112, sum_135, mul_1103, sum_133, mul_1094, sum_131, mul_1082, sum_129, mul_1073, sum_127, mul_1064, sum_125, mul_1052, sum_123, mul_1043, sum_121, mul_1034, sum_119, mul_1022, sum_117, mul_1000, sum_112, mul_991, sum_110, mul_979, sum_108, mul_957, sum_103, mul_948, sum_101, mul_936, sum_99, mul_914, sum_94, mul_905, sum_92, mul_893, sum_90, mul_871, sum_85, mul_862, sum_83, mul_850, sum_81, mul_828, sum_76, mul_819, sum_74, mul_807, sum_72, mul_785, sum_67, mul_776, sum_65, mul_764, sum_63, mul_742, sum_58, mul_733, sum_56, mul_721, sum_54, mul_699, sum_49, mul_690, sum_47, mul_678, sum_45, mul_656, sum_40, mul_647, sum_38, mul_635, sum_36, mul_613, sum_31, mul_604, sum_29, mul_592, sum_27, mul_570, sum_22, mul_561, sum_20, mul_549, sum_18, mul_527, sum_13, mul_518, sum_11, mul_506, sum_9, mul_484, sum_4, mul_475, sum_2, getitem_347, getitem_344, getitem_341, getitem_338, getitem_335, getitem_332, getitem_329, getitem_326, getitem_323, getitem_320, getitem_317, getitem_314, getitem_315, mul_1013, sum_115, getitem_311, getitem_312, getitem_308, getitem_305, getitem_302, getitem_299, getitem_300, mul_970, sum_106, getitem_296, getitem_297, getitem_293, getitem_290, getitem_287, getitem_284, getitem_285, mul_927, sum_97, getitem_281, getitem_282, getitem_278, getitem_275, getitem_272, getitem_269, getitem_270, mul_884, sum_88, getitem_266, getitem_267, getitem_263, getitem_260, getitem_257, getitem_254, getitem_255, mul_841, sum_79, getitem_251, getitem_252, getitem_248, getitem_245, getitem_242, getitem_239, getitem_240, mul_798, sum_70, getitem_236, getitem_237, getitem_233, getitem_230, getitem_227, getitem_224, getitem_225, mul_755, sum_61, getitem_221, getitem_222, getitem_218, getitem_215, getitem_212, getitem_209, getitem_210, mul_712, sum_52, getitem_206, getitem_207, getitem_203, getitem_200, getitem_197, getitem_194, getitem_195, mul_669, sum_43, getitem_191, getitem_192, getitem_188, getitem_185, getitem_182, getitem_179, getitem_180, mul_626, sum_34, getitem_176, getitem_177, getitem_173, getitem_170, getitem_167, getitem_164, getitem_165, mul_583, sum_25, getitem_161, getitem_162, getitem_158, getitem_155, getitem_152, getitem_149, getitem_150, mul_540, sum_16, getitem_146, getitem_147, getitem_143, getitem_140, getitem_137, getitem_134, getitem_135, mul_497, sum_7, getitem_131, getitem_132, getitem_128, getitem_125, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    