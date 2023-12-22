from __future__ import annotations



def forward(self, primals_1: "f32[128]", primals_3: "f32[128]", primals_5: "f32[128]", primals_6: "f32[128]", primals_8: "f32[128]", primals_9: "f32[128]", primals_11: "f32[128]", primals_12: "f32[128]", primals_14: "f32[256]", primals_16: "f32[256]", primals_17: "f32[256]", primals_19: "f32[256]", primals_20: "f32[256]", primals_22: "f32[256]", primals_23: "f32[256]", primals_25: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_39: "f32[512]", primals_40: "f32[512]", primals_42: "f32[512]", primals_43: "f32[512]", primals_45: "f32[512]", primals_46: "f32[512]", primals_48: "f32[512]", primals_49: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512]", primals_54: "f32[512]", primals_55: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512]", primals_60: "f32[512]", primals_61: "f32[512]", primals_63: "f32[512]", primals_64: "f32[512]", primals_66: "f32[512]", primals_67: "f32[512]", primals_69: "f32[512]", primals_70: "f32[512]", primals_72: "f32[512]", primals_73: "f32[512]", primals_75: "f32[512]", primals_76: "f32[512]", primals_78: "f32[512]", primals_79: "f32[512]", primals_81: "f32[512]", primals_82: "f32[512]", primals_84: "f32[512]", primals_85: "f32[512]", primals_87: "f32[512]", primals_88: "f32[512]", primals_90: "f32[512]", primals_91: "f32[512]", primals_93: "f32[512]", primals_94: "f32[512]", primals_96: "f32[512]", primals_97: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_102: "f32[512]", primals_103: "f32[512]", primals_105: "f32[512]", primals_106: "f32[512]", primals_108: "f32[1024]", primals_110: "f32[1024]", primals_111: "f32[1024]", primals_113: "f32[1024]", primals_114: "f32[1024]", primals_116: "f32[1024]", primals_117: "f32[1024]", primals_119: "f32[128, 3, 4, 4]", primals_121: "f32[128, 1, 7, 7]", primals_127: "f32[128, 1, 7, 7]", primals_133: "f32[128, 1, 7, 7]", primals_139: "f32[256, 128, 2, 2]", primals_141: "f32[256, 1, 7, 7]", primals_147: "f32[256, 1, 7, 7]", primals_153: "f32[256, 1, 7, 7]", primals_159: "f32[512, 256, 2, 2]", primals_161: "f32[512, 1, 7, 7]", primals_167: "f32[512, 1, 7, 7]", primals_173: "f32[512, 1, 7, 7]", primals_179: "f32[512, 1, 7, 7]", primals_185: "f32[512, 1, 7, 7]", primals_191: "f32[512, 1, 7, 7]", primals_197: "f32[512, 1, 7, 7]", primals_203: "f32[512, 1, 7, 7]", primals_209: "f32[512, 1, 7, 7]", primals_215: "f32[512, 1, 7, 7]", primals_221: "f32[512, 1, 7, 7]", primals_227: "f32[512, 1, 7, 7]", primals_233: "f32[512, 1, 7, 7]", primals_239: "f32[512, 1, 7, 7]", primals_245: "f32[512, 1, 7, 7]", primals_251: "f32[512, 1, 7, 7]", primals_257: "f32[512, 1, 7, 7]", primals_263: "f32[512, 1, 7, 7]", primals_269: "f32[512, 1, 7, 7]", primals_275: "f32[512, 1, 7, 7]", primals_281: "f32[512, 1, 7, 7]", primals_287: "f32[512, 1, 7, 7]", primals_293: "f32[512, 1, 7, 7]", primals_299: "f32[512, 1, 7, 7]", primals_305: "f32[512, 1, 7, 7]", primals_311: "f32[512, 1, 7, 7]", primals_317: "f32[512, 1, 7, 7]", primals_323: "f32[1024, 512, 2, 2]", primals_325: "f32[1024, 1, 7, 7]", primals_331: "f32[1024, 1, 7, 7]", primals_337: "f32[1024, 1, 7, 7]", primals_345: "f32[8, 3, 224, 224]", mul: "f32[8, 56, 56, 128]", permute_1: "f32[8, 128, 56, 56]", convolution_1: "f32[8, 128, 56, 56]", getitem_3: "f32[8, 56, 56, 1]", rsqrt_1: "f32[8, 56, 56, 1]", view: "f32[25088, 128]", addmm: "f32[25088, 512]", view_2: "f32[25088, 512]", addmm_1: "f32[25088, 128]", add_5: "f32[8, 128, 56, 56]", convolution_2: "f32[8, 128, 56, 56]", getitem_5: "f32[8, 56, 56, 1]", rsqrt_2: "f32[8, 56, 56, 1]", view_5: "f32[25088, 128]", addmm_2: "f32[25088, 512]", view_7: "f32[25088, 512]", addmm_3: "f32[25088, 128]", add_9: "f32[8, 128, 56, 56]", convolution_3: "f32[8, 128, 56, 56]", getitem_7: "f32[8, 56, 56, 1]", rsqrt_3: "f32[8, 56, 56, 1]", view_10: "f32[25088, 128]", addmm_4: "f32[25088, 512]", view_12: "f32[25088, 512]", addmm_5: "f32[25088, 128]", mul_20: "f32[8, 56, 56, 128]", permute_15: "f32[8, 128, 56, 56]", convolution_4: "f32[8, 256, 28, 28]", convolution_5: "f32[8, 256, 28, 28]", getitem_11: "f32[8, 28, 28, 1]", rsqrt_5: "f32[8, 28, 28, 1]", view_15: "f32[6272, 256]", addmm_6: "f32[6272, 1024]", view_17: "f32[6272, 1024]", addmm_7: "f32[6272, 256]", add_19: "f32[8, 256, 28, 28]", convolution_6: "f32[8, 256, 28, 28]", getitem_13: "f32[8, 28, 28, 1]", rsqrt_6: "f32[8, 28, 28, 1]", view_20: "f32[6272, 256]", addmm_8: "f32[6272, 1024]", view_22: "f32[6272, 1024]", addmm_9: "f32[6272, 256]", add_23: "f32[8, 256, 28, 28]", convolution_7: "f32[8, 256, 28, 28]", getitem_15: "f32[8, 28, 28, 1]", rsqrt_7: "f32[8, 28, 28, 1]", view_25: "f32[6272, 256]", addmm_10: "f32[6272, 1024]", view_27: "f32[6272, 1024]", addmm_11: "f32[6272, 256]", mul_40: "f32[8, 28, 28, 256]", permute_29: "f32[8, 256, 28, 28]", convolution_8: "f32[8, 512, 14, 14]", convolution_9: "f32[8, 512, 14, 14]", getitem_19: "f32[8, 14, 14, 1]", rsqrt_9: "f32[8, 14, 14, 1]", view_30: "f32[1568, 512]", addmm_12: "f32[1568, 2048]", view_32: "f32[1568, 2048]", addmm_13: "f32[1568, 512]", add_33: "f32[8, 512, 14, 14]", convolution_10: "f32[8, 512, 14, 14]", getitem_21: "f32[8, 14, 14, 1]", rsqrt_10: "f32[8, 14, 14, 1]", view_35: "f32[1568, 512]", addmm_14: "f32[1568, 2048]", view_37: "f32[1568, 2048]", addmm_15: "f32[1568, 512]", add_37: "f32[8, 512, 14, 14]", convolution_11: "f32[8, 512, 14, 14]", getitem_23: "f32[8, 14, 14, 1]", rsqrt_11: "f32[8, 14, 14, 1]", view_40: "f32[1568, 512]", addmm_16: "f32[1568, 2048]", view_42: "f32[1568, 2048]", addmm_17: "f32[1568, 512]", add_41: "f32[8, 512, 14, 14]", convolution_12: "f32[8, 512, 14, 14]", getitem_25: "f32[8, 14, 14, 1]", rsqrt_12: "f32[8, 14, 14, 1]", view_45: "f32[1568, 512]", addmm_18: "f32[1568, 2048]", view_47: "f32[1568, 2048]", addmm_19: "f32[1568, 512]", add_45: "f32[8, 512, 14, 14]", convolution_13: "f32[8, 512, 14, 14]", getitem_27: "f32[8, 14, 14, 1]", rsqrt_13: "f32[8, 14, 14, 1]", view_50: "f32[1568, 512]", addmm_20: "f32[1568, 2048]", view_52: "f32[1568, 2048]", addmm_21: "f32[1568, 512]", add_49: "f32[8, 512, 14, 14]", convolution_14: "f32[8, 512, 14, 14]", getitem_29: "f32[8, 14, 14, 1]", rsqrt_14: "f32[8, 14, 14, 1]", view_55: "f32[1568, 512]", addmm_22: "f32[1568, 2048]", view_57: "f32[1568, 2048]", addmm_23: "f32[1568, 512]", add_53: "f32[8, 512, 14, 14]", convolution_15: "f32[8, 512, 14, 14]", getitem_31: "f32[8, 14, 14, 1]", rsqrt_15: "f32[8, 14, 14, 1]", view_60: "f32[1568, 512]", addmm_24: "f32[1568, 2048]", view_62: "f32[1568, 2048]", addmm_25: "f32[1568, 512]", add_57: "f32[8, 512, 14, 14]", convolution_16: "f32[8, 512, 14, 14]", getitem_33: "f32[8, 14, 14, 1]", rsqrt_16: "f32[8, 14, 14, 1]", view_65: "f32[1568, 512]", addmm_26: "f32[1568, 2048]", view_67: "f32[1568, 2048]", addmm_27: "f32[1568, 512]", add_61: "f32[8, 512, 14, 14]", convolution_17: "f32[8, 512, 14, 14]", getitem_35: "f32[8, 14, 14, 1]", rsqrt_17: "f32[8, 14, 14, 1]", view_70: "f32[1568, 512]", addmm_28: "f32[1568, 2048]", view_72: "f32[1568, 2048]", addmm_29: "f32[1568, 512]", add_65: "f32[8, 512, 14, 14]", convolution_18: "f32[8, 512, 14, 14]", getitem_37: "f32[8, 14, 14, 1]", rsqrt_18: "f32[8, 14, 14, 1]", view_75: "f32[1568, 512]", addmm_30: "f32[1568, 2048]", view_77: "f32[1568, 2048]", addmm_31: "f32[1568, 512]", add_69: "f32[8, 512, 14, 14]", convolution_19: "f32[8, 512, 14, 14]", getitem_39: "f32[8, 14, 14, 1]", rsqrt_19: "f32[8, 14, 14, 1]", view_80: "f32[1568, 512]", addmm_32: "f32[1568, 2048]", view_82: "f32[1568, 2048]", addmm_33: "f32[1568, 512]", add_73: "f32[8, 512, 14, 14]", convolution_20: "f32[8, 512, 14, 14]", getitem_41: "f32[8, 14, 14, 1]", rsqrt_20: "f32[8, 14, 14, 1]", view_85: "f32[1568, 512]", addmm_34: "f32[1568, 2048]", view_87: "f32[1568, 2048]", addmm_35: "f32[1568, 512]", add_77: "f32[8, 512, 14, 14]", convolution_21: "f32[8, 512, 14, 14]", getitem_43: "f32[8, 14, 14, 1]", rsqrt_21: "f32[8, 14, 14, 1]", view_90: "f32[1568, 512]", addmm_36: "f32[1568, 2048]", view_92: "f32[1568, 2048]", addmm_37: "f32[1568, 512]", add_81: "f32[8, 512, 14, 14]", convolution_22: "f32[8, 512, 14, 14]", getitem_45: "f32[8, 14, 14, 1]", rsqrt_22: "f32[8, 14, 14, 1]", view_95: "f32[1568, 512]", addmm_38: "f32[1568, 2048]", view_97: "f32[1568, 2048]", addmm_39: "f32[1568, 512]", add_85: "f32[8, 512, 14, 14]", convolution_23: "f32[8, 512, 14, 14]", getitem_47: "f32[8, 14, 14, 1]", rsqrt_23: "f32[8, 14, 14, 1]", view_100: "f32[1568, 512]", addmm_40: "f32[1568, 2048]", view_102: "f32[1568, 2048]", addmm_41: "f32[1568, 512]", add_89: "f32[8, 512, 14, 14]", convolution_24: "f32[8, 512, 14, 14]", getitem_49: "f32[8, 14, 14, 1]", rsqrt_24: "f32[8, 14, 14, 1]", view_105: "f32[1568, 512]", addmm_42: "f32[1568, 2048]", view_107: "f32[1568, 2048]", addmm_43: "f32[1568, 512]", add_93: "f32[8, 512, 14, 14]", convolution_25: "f32[8, 512, 14, 14]", getitem_51: "f32[8, 14, 14, 1]", rsqrt_25: "f32[8, 14, 14, 1]", view_110: "f32[1568, 512]", addmm_44: "f32[1568, 2048]", view_112: "f32[1568, 2048]", addmm_45: "f32[1568, 512]", add_97: "f32[8, 512, 14, 14]", convolution_26: "f32[8, 512, 14, 14]", getitem_53: "f32[8, 14, 14, 1]", rsqrt_26: "f32[8, 14, 14, 1]", view_115: "f32[1568, 512]", addmm_46: "f32[1568, 2048]", view_117: "f32[1568, 2048]", addmm_47: "f32[1568, 512]", add_101: "f32[8, 512, 14, 14]", convolution_27: "f32[8, 512, 14, 14]", getitem_55: "f32[8, 14, 14, 1]", rsqrt_27: "f32[8, 14, 14, 1]", view_120: "f32[1568, 512]", addmm_48: "f32[1568, 2048]", view_122: "f32[1568, 2048]", addmm_49: "f32[1568, 512]", add_105: "f32[8, 512, 14, 14]", convolution_28: "f32[8, 512, 14, 14]", getitem_57: "f32[8, 14, 14, 1]", rsqrt_28: "f32[8, 14, 14, 1]", view_125: "f32[1568, 512]", addmm_50: "f32[1568, 2048]", view_127: "f32[1568, 2048]", addmm_51: "f32[1568, 512]", add_109: "f32[8, 512, 14, 14]", convolution_29: "f32[8, 512, 14, 14]", getitem_59: "f32[8, 14, 14, 1]", rsqrt_29: "f32[8, 14, 14, 1]", view_130: "f32[1568, 512]", addmm_52: "f32[1568, 2048]", view_132: "f32[1568, 2048]", addmm_53: "f32[1568, 512]", add_113: "f32[8, 512, 14, 14]", convolution_30: "f32[8, 512, 14, 14]", getitem_61: "f32[8, 14, 14, 1]", rsqrt_30: "f32[8, 14, 14, 1]", view_135: "f32[1568, 512]", addmm_54: "f32[1568, 2048]", view_137: "f32[1568, 2048]", addmm_55: "f32[1568, 512]", add_117: "f32[8, 512, 14, 14]", convolution_31: "f32[8, 512, 14, 14]", getitem_63: "f32[8, 14, 14, 1]", rsqrt_31: "f32[8, 14, 14, 1]", view_140: "f32[1568, 512]", addmm_56: "f32[1568, 2048]", view_142: "f32[1568, 2048]", addmm_57: "f32[1568, 512]", add_121: "f32[8, 512, 14, 14]", convolution_32: "f32[8, 512, 14, 14]", getitem_65: "f32[8, 14, 14, 1]", rsqrt_32: "f32[8, 14, 14, 1]", view_145: "f32[1568, 512]", addmm_58: "f32[1568, 2048]", view_147: "f32[1568, 2048]", addmm_59: "f32[1568, 512]", add_125: "f32[8, 512, 14, 14]", convolution_33: "f32[8, 512, 14, 14]", getitem_67: "f32[8, 14, 14, 1]", rsqrt_33: "f32[8, 14, 14, 1]", view_150: "f32[1568, 512]", addmm_60: "f32[1568, 2048]", view_152: "f32[1568, 2048]", addmm_61: "f32[1568, 512]", add_129: "f32[8, 512, 14, 14]", convolution_34: "f32[8, 512, 14, 14]", getitem_69: "f32[8, 14, 14, 1]", rsqrt_34: "f32[8, 14, 14, 1]", view_155: "f32[1568, 512]", addmm_62: "f32[1568, 2048]", view_157: "f32[1568, 2048]", addmm_63: "f32[1568, 512]", add_133: "f32[8, 512, 14, 14]", convolution_35: "f32[8, 512, 14, 14]", getitem_71: "f32[8, 14, 14, 1]", rsqrt_35: "f32[8, 14, 14, 1]", view_160: "f32[1568, 512]", addmm_64: "f32[1568, 2048]", view_162: "f32[1568, 2048]", addmm_65: "f32[1568, 512]", mul_204: "f32[8, 14, 14, 512]", permute_139: "f32[8, 512, 14, 14]", convolution_36: "f32[8, 1024, 7, 7]", convolution_37: "f32[8, 1024, 7, 7]", getitem_75: "f32[8, 7, 7, 1]", rsqrt_37: "f32[8, 7, 7, 1]", view_165: "f32[392, 1024]", addmm_66: "f32[392, 4096]", view_167: "f32[392, 4096]", addmm_67: "f32[392, 1024]", add_143: "f32[8, 1024, 7, 7]", convolution_38: "f32[8, 1024, 7, 7]", getitem_77: "f32[8, 7, 7, 1]", rsqrt_38: "f32[8, 7, 7, 1]", view_170: "f32[392, 1024]", addmm_68: "f32[392, 4096]", view_172: "f32[392, 4096]", addmm_69: "f32[392, 1024]", add_147: "f32[8, 1024, 7, 7]", convolution_39: "f32[8, 1024, 7, 7]", getitem_79: "f32[8, 7, 7, 1]", rsqrt_39: "f32[8, 7, 7, 1]", view_175: "f32[392, 1024]", addmm_70: "f32[392, 4096]", view_177: "f32[392, 4096]", addmm_71: "f32[392, 1024]", mul_224: "f32[8, 1, 1, 1024]", clone_73: "f32[8, 1024]", permute_155: "f32[1000, 1024]", div: "f32[8, 1, 1, 1]", permute_162: "f32[1024, 4096]", permute_166: "f32[4096, 1024]", permute_172: "f32[1024, 4096]", permute_176: "f32[4096, 1024]", permute_182: "f32[1024, 4096]", permute_186: "f32[4096, 1024]", div_5: "f32[8, 14, 14, 1]", permute_194: "f32[512, 2048]", permute_198: "f32[2048, 512]", permute_204: "f32[512, 2048]", permute_208: "f32[2048, 512]", permute_214: "f32[512, 2048]", permute_218: "f32[2048, 512]", permute_224: "f32[512, 2048]", permute_228: "f32[2048, 512]", permute_234: "f32[512, 2048]", permute_238: "f32[2048, 512]", permute_244: "f32[512, 2048]", permute_248: "f32[2048, 512]", permute_254: "f32[512, 2048]", permute_258: "f32[2048, 512]", permute_264: "f32[512, 2048]", permute_268: "f32[2048, 512]", permute_274: "f32[512, 2048]", permute_278: "f32[2048, 512]", permute_284: "f32[512, 2048]", permute_288: "f32[2048, 512]", permute_294: "f32[512, 2048]", permute_298: "f32[2048, 512]", permute_304: "f32[512, 2048]", permute_308: "f32[2048, 512]", permute_314: "f32[512, 2048]", permute_318: "f32[2048, 512]", permute_324: "f32[512, 2048]", permute_328: "f32[2048, 512]", permute_334: "f32[512, 2048]", permute_338: "f32[2048, 512]", permute_344: "f32[512, 2048]", permute_348: "f32[2048, 512]", permute_354: "f32[512, 2048]", permute_358: "f32[2048, 512]", permute_364: "f32[512, 2048]", permute_368: "f32[2048, 512]", permute_374: "f32[512, 2048]", permute_378: "f32[2048, 512]", permute_384: "f32[512, 2048]", permute_388: "f32[2048, 512]", permute_394: "f32[512, 2048]", permute_398: "f32[2048, 512]", permute_404: "f32[512, 2048]", permute_408: "f32[2048, 512]", permute_414: "f32[512, 2048]", permute_418: "f32[2048, 512]", permute_424: "f32[512, 2048]", permute_428: "f32[2048, 512]", permute_434: "f32[512, 2048]", permute_438: "f32[2048, 512]", permute_444: "f32[512, 2048]", permute_448: "f32[2048, 512]", permute_454: "f32[512, 2048]", permute_458: "f32[2048, 512]", div_33: "f32[8, 28, 28, 1]", permute_466: "f32[256, 1024]", permute_470: "f32[1024, 256]", permute_476: "f32[256, 1024]", permute_480: "f32[1024, 256]", permute_486: "f32[256, 1024]", permute_490: "f32[1024, 256]", div_37: "f32[8, 56, 56, 1]", permute_498: "f32[128, 512]", permute_502: "f32[512, 128]", permute_508: "f32[128, 512]", permute_512: "f32[512, 128]", permute_518: "f32[128, 512]", permute_522: "f32[512, 128]", div_41: "f32[8, 56, 56, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_2: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_1: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_2, getitem_3);  permute_2 = getitem_3 = None
    mul_2: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_1: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(addmm, [8, 56, 56, 512]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_5: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, 0.7071067811865476)
    erf: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_4: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_3: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(addmm_1, [8, 56, 56, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_5: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_3, [0, 3, 1, 2]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_4: "f32[1, 128, 1, 1]" = torch.ops.aten.reshape.default(primals_5, [1, -1, 1, 1]);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_6: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_2, [0, 2, 3, 1]);  convolution_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_2: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_6, getitem_5);  permute_6 = getitem_5 = None
    mul_8: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_6: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(addmm_2, [8, 56, 56, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_11: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476)
    erf_1: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_8: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_8: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(addmm_3, [8, 56, 56, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_9: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_8, [0, 3, 1, 2]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_9: "f32[1, 128, 1, 1]" = torch.ops.aten.reshape.default(primals_8, [1, -1, 1, 1]);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_10: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(convolution_3, [0, 2, 3, 1]);  convolution_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_3: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(permute_10, getitem_7);  permute_10 = getitem_7 = None
    mul_14: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_11: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(addmm_4, [8, 56, 56, 512]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_17: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, 0.7071067811865476)
    erf_2: "f32[8, 56, 56, 512]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
    add_12: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_13: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(addmm_5, [8, 56, 56, 128]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_13: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(view_13, [0, 3, 1, 2]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_14: "f32[1, 128, 1, 1]" = torch.ops.aten.reshape.default(primals_11, [1, -1, 1, 1]);  primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_16: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_5, [0, 2, 3, 1]);  convolution_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_5: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_16, getitem_11);  permute_16 = getitem_11 = None
    mul_22: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_16: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(addmm_6, [8, 28, 28, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_25: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf_3: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_18: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_18: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(addmm_7, [8, 28, 28, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_19: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_18, [0, 3, 1, 2]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_19: "f32[1, 256, 1, 1]" = torch.ops.aten.reshape.default(primals_16, [1, -1, 1, 1]);  primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_20: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_6, [0, 2, 3, 1]);  convolution_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_6: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_20, getitem_13);  permute_20 = getitem_13 = None
    mul_28: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(addmm_8, [8, 28, 28, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_31: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf_4: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_22: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(addmm_9, [8, 28, 28, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_23: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_23, [0, 3, 1, 2]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_24: "f32[1, 256, 1, 1]" = torch.ops.aten.reshape.default(primals_19, [1, -1, 1, 1]);  primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_24: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(convolution_7, [0, 2, 3, 1]);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_7: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(permute_24, getitem_15);  permute_24 = getitem_15 = None
    mul_34: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(addmm_10, [8, 28, 28, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_37: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
    erf_5: "f32[8, 28, 28, 1024]" = torch.ops.aten.erf.default(mul_37);  mul_37 = None
    add_26: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_28: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(addmm_11, [8, 28, 28, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_27: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(view_28, [0, 3, 1, 2]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_29: "f32[1, 256, 1, 1]" = torch.ops.aten.reshape.default(primals_22, [1, -1, 1, 1]);  primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_30: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_9, [0, 2, 3, 1]);  convolution_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_9: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_30, getitem_19);  permute_30 = getitem_19 = None
    mul_42: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_12, [8, 14, 14, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_45: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, 0.7071067811865476)
    erf_6: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_33: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_13, [8, 14, 14, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_33: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_33, [0, 3, 1, 2]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_34: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_27, [1, -1, 1, 1]);  primals_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_34: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_10, [0, 2, 3, 1]);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_10: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_34, getitem_21);  permute_34 = getitem_21 = None
    mul_48: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_14, [8, 14, 14, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, 0.7071067811865476)
    erf_7: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_36: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_38: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_15, [8, 14, 14, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_37: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_38, [0, 3, 1, 2]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_39: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_30, [1, -1, 1, 1]);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_38: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_11: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_38, getitem_23);  permute_38 = getitem_23 = None
    mul_54: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_41: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_16, [8, 14, 14, 2048]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_57: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_8: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
    add_40: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_17, [8, 14, 14, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_41: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_43, [0, 3, 1, 2]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_44: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_33, [1, -1, 1, 1]);  primals_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_42: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_12, [0, 2, 3, 1]);  convolution_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_12: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_42, getitem_25);  permute_42 = getitem_25 = None
    mul_60: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_18, [8, 14, 14, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_63: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_9: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_44: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_48: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_19, [8, 14, 14, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_45: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_48, [0, 3, 1, 2]);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_49: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_36, [1, -1, 1, 1]);  primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_46: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_13, [0, 2, 3, 1]);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_13: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_46, getitem_27);  permute_46 = getitem_27 = None
    mul_66: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_20, [8, 14, 14, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_69: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_10: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_48: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_21, [8, 14, 14, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_49: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_53, [0, 3, 1, 2]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_54: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_39, [1, -1, 1, 1]);  primals_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_50: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_14, [0, 2, 3, 1]);  convolution_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_14: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_50, getitem_29);  permute_50 = getitem_29 = None
    mul_72: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_22, [8, 14, 14, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_75: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
    erf_11: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_52: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_58: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_23, [8, 14, 14, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_53: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_58, [0, 3, 1, 2]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_59: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_42, [1, -1, 1, 1]);  primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_54: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_15: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_54, getitem_31);  permute_54 = getitem_31 = None
    mul_78: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_61: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_24, [8, 14, 14, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, 0.7071067811865476)
    erf_12: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_56: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_63: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_25, [8, 14, 14, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_57: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_63, [0, 3, 1, 2]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_64: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_45, [1, -1, 1, 1]);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_58: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_16, [0, 2, 3, 1]);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_16: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_58, getitem_33);  permute_58 = getitem_33 = None
    mul_84: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_26, [8, 14, 14, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_87: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, 0.7071067811865476)
    erf_13: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_60: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_68: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_27, [8, 14, 14, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_61: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_68, [0, 3, 1, 2]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_69: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_48, [1, -1, 1, 1]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_62: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_17, [0, 2, 3, 1]);  convolution_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_17: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_62, getitem_35);  permute_62 = getitem_35 = None
    mul_90: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_71: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_28, [8, 14, 14, 2048]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_93: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, 0.7071067811865476)
    erf_14: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_64: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_73: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_29, [8, 14, 14, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_65: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_73, [0, 3, 1, 2]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_74: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_51, [1, -1, 1, 1]);  primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_66: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_18: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_66, getitem_37);  permute_66 = getitem_37 = None
    mul_96: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_30, [8, 14, 14, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_99: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_15: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_68: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_78: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_31, [8, 14, 14, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_69: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_78, [0, 3, 1, 2]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_79: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_54, [1, -1, 1, 1]);  primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_70: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_19, [0, 2, 3, 1]);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_19: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_70, getitem_39);  permute_70 = getitem_39 = None
    mul_102: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_32, [8, 14, 14, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_105: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, 0.7071067811865476)
    erf_16: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_72: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_33, [8, 14, 14, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_73: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_83, [0, 3, 1, 2]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_84: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_57, [1, -1, 1, 1]);  primals_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_74: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_20, [0, 2, 3, 1]);  convolution_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_20: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_74, getitem_41);  permute_74 = getitem_41 = None
    mul_108: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_34, [8, 14, 14, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_17: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_76: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_88: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_35, [8, 14, 14, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_77: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_88, [0, 3, 1, 2]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_89: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_60, [1, -1, 1, 1]);  primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_78: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_21: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_78, getitem_43);  permute_78 = getitem_43 = None
    mul_114: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_36, [8, 14, 14, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_117: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, 0.7071067811865476)
    erf_18: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
    add_80: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_37, [8, 14, 14, 512]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_81: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_93, [0, 3, 1, 2]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_94: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_63, [1, -1, 1, 1]);  primals_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_82: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_22, [0, 2, 3, 1]);  convolution_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_22: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_82, getitem_45);  permute_82 = getitem_45 = None
    mul_120: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_38, [8, 14, 14, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_123: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, 0.7071067811865476)
    erf_19: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_84: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_98: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_39, [8, 14, 14, 512]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_85: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_98, [0, 3, 1, 2]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_99: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_66, [1, -1, 1, 1]);  primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_86: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_23, [0, 2, 3, 1]);  convolution_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_23: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_86, getitem_47);  permute_86 = getitem_47 = None
    mul_126: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_40, [8, 14, 14, 2048]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_129: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, 0.7071067811865476)
    erf_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_88: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_103: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_41, [8, 14, 14, 512]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_89: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_103, [0, 3, 1, 2]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_104: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_69, [1, -1, 1, 1]);  primals_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_90: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_24: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_90, getitem_49);  permute_90 = getitem_49 = None
    mul_132: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_42, [8, 14, 14, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_135: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_21: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_92: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_43, [8, 14, 14, 512]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_93: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_108, [0, 3, 1, 2]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_109: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_72, [1, -1, 1, 1]);  primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_94: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_25, [0, 2, 3, 1]);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_25: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_94, getitem_51);  permute_94 = getitem_51 = None
    mul_138: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_111: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_44, [8, 14, 14, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, 0.7071067811865476)
    erf_22: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_96: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_113: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_45, [8, 14, 14, 512]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_97: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_113, [0, 3, 1, 2]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_114: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_75, [1, -1, 1, 1]);  primals_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_98: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_26, [0, 2, 3, 1]);  convolution_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_26: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_98, getitem_53);  permute_98 = getitem_53 = None
    mul_144: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_46, [8, 14, 14, 2048]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_147: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_100: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_47, [8, 14, 14, 512]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_101: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_118, [0, 3, 1, 2]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_119: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_78, [1, -1, 1, 1]);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_102: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_27: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_102, getitem_55);  permute_102 = getitem_55 = None
    mul_150: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_121: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_48, [8, 14, 14, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_153: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, 0.7071067811865476)
    erf_24: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_153);  mul_153 = None
    add_104: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_123: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_49, [8, 14, 14, 512]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_105: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_123, [0, 3, 1, 2]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_124: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_81, [1, -1, 1, 1]);  primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_106: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_28, [0, 2, 3, 1]);  convolution_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_28: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_106, getitem_57);  permute_106 = getitem_57 = None
    mul_156: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_50, [8, 14, 14, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_159: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476)
    erf_25: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_108: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_51, [8, 14, 14, 512]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_109: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_128, [0, 3, 1, 2]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_129: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_84, [1, -1, 1, 1]);  primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_110: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_29, [0, 2, 3, 1]);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_29: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_110, getitem_59);  permute_110 = getitem_59 = None
    mul_162: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_131: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_52, [8, 14, 14, 2048]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_165: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_112: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_133: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_53, [8, 14, 14, 512]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_113: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_133, [0, 3, 1, 2]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_134: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_87, [1, -1, 1, 1]);  primals_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_114: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_30: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_114, getitem_61);  permute_114 = getitem_61 = None
    mul_168: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_54, [8, 14, 14, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_171: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476)
    erf_27: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_116: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_138: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_55, [8, 14, 14, 512]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_117: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_138, [0, 3, 1, 2]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_139: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_90, [1, -1, 1, 1]);  primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_118: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_31, [0, 2, 3, 1]);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_31: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_118, getitem_63);  permute_118 = getitem_63 = None
    mul_174: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_56, [8, 14, 14, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_177: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_28: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_120: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_57, [8, 14, 14, 512]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_121: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_143, [0, 3, 1, 2]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_144: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_93, [1, -1, 1, 1]);  primals_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_122: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_32, [0, 2, 3, 1]);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_32: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_122, getitem_65);  permute_122 = getitem_65 = None
    mul_180: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_146: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_58, [8, 14, 14, 2048]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_183: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_124: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_148: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_59, [8, 14, 14, 512]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_125: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_148, [0, 3, 1, 2]);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_149: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_96, [1, -1, 1, 1]);  primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_126: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_33: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_126, getitem_67);  permute_126 = getitem_67 = None
    mul_186: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_151: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_60, [8, 14, 14, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_189: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_30: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_189);  mul_189 = None
    add_128: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_153: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_61, [8, 14, 14, 512]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_129: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_153, [0, 3, 1, 2]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_154: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_99, [1, -1, 1, 1]);  primals_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_130: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_34, [0, 2, 3, 1]);  convolution_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_34: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_130, getitem_69);  permute_130 = getitem_69 = None
    mul_192: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_156: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_62, [8, 14, 14, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_195: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, 0.7071067811865476)
    erf_31: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_132: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_158: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_63, [8, 14, 14, 512]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_133: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_158, [0, 3, 1, 2]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_159: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_102, [1, -1, 1, 1]);  primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_134: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(convolution_35, [0, 2, 3, 1]);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_35: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(permute_134, getitem_71);  permute_134 = getitem_71 = None
    mul_198: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_161: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(addmm_64, [8, 14, 14, 2048]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_201: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, 0.7071067811865476)
    erf_32: "f32[8, 14, 14, 2048]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_136: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_163: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(addmm_65, [8, 14, 14, 512]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_137: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(view_163, [0, 3, 1, 2]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_164: "f32[1, 512, 1, 1]" = torch.ops.aten.reshape.default(primals_105, [1, -1, 1, 1]);  primals_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_140: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_37, [0, 2, 3, 1]);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_37: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(permute_140, getitem_75);  permute_140 = getitem_75 = None
    mul_206: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(addmm_66, [8, 7, 7, 4096]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_209: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, 0.7071067811865476)
    erf_33: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_142: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(addmm_67, [8, 7, 7, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_143: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(view_168, [0, 3, 1, 2]);  view_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.reshape.default(primals_110, [1, -1, 1, 1]);  primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_144: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_38, [0, 2, 3, 1]);  convolution_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_38: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(permute_144, getitem_77);  permute_144 = getitem_77 = None
    mul_212: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_171: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(addmm_68, [8, 7, 7, 4096]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_215: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, 0.7071067811865476)
    erf_34: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
    add_146: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_173: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(addmm_69, [8, 7, 7, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_147: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(view_173, [0, 3, 1, 2]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_174: "f32[1, 1024, 1, 1]" = torch.ops.aten.reshape.default(primals_113, [1, -1, 1, 1]);  primals_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_148: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(convolution_39, [0, 2, 3, 1]);  convolution_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_39: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(permute_148, getitem_79);  permute_148 = getitem_79 = None
    mul_218: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_176: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(addmm_70, [8, 7, 7, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_221: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_35: "f32[8, 7, 7, 4096]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_150: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_178: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(addmm_71, [8, 7, 7, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_151: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(view_178, [0, 3, 1, 2]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    view_179: "f32[1, 1024, 1, 1]" = torch.ops.aten.reshape.default(primals_116, [1, -1, 1, 1]);  primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:207, code: x = self.fc(x)
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_155);  permute_155 = None
    permute_156: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_156, clone_73);  permute_156 = clone_73 = None
    permute_157: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_181: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_158: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:202, code: x = self.flatten(x)
    view_182: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_159: "f32[8, 1, 1, 1024]" = torch.ops.aten.permute.default(view_182, [0, 2, 3, 1]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_227: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(permute_159, primals_117);  primals_117 = None
    mul_228: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_227, 1024)
    sum_2: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [3], True)
    mul_229: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_227, mul_224);  mul_227 = None
    sum_3: "f32[8, 1, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [3], True);  mul_229 = None
    mul_230: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_224, sum_3);  sum_3 = None
    sub_42: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_228, sum_2);  mul_228 = sum_2 = None
    sub_43: "f32[8, 1, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_42, mul_230);  sub_42 = mul_230 = None
    mul_231: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(div, sub_43);  div = sub_43 = None
    mul_232: "f32[8, 1, 1, 1024]" = torch.ops.aten.mul.Tensor(permute_159, mul_224);  mul_224 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1, 2]);  mul_232 = None
    sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(permute_159, [0, 1, 2]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_160: "f32[8, 1024, 1, 1]" = torch.ops.aten.permute.default(mul_231, [0, 3, 1, 2]);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    squeeze: "f32[8, 1024, 1]" = torch.ops.aten.squeeze.dim(permute_160, 3);  permute_160 = None
    squeeze_1: "f32[8, 1024]" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
    full: "f32[8192]" = torch.ops.aten.full.default([8192], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    as_strided_1: "f32[8, 1024]" = torch.ops.aten.as_strided.default(full, [8, 1024], [1024, 1], 0)
    as_strided_scatter: "f32[8192]" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, [8, 1024], [1024, 1], 0);  full = squeeze_1 = None
    as_strided_4: "f32[8, 1024, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, [8, 1024, 1, 1], [1024, 1, 1, 1], 0);  as_strided_scatter = None
    expand_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.expand.default(as_strided_4, [8, 1024, 7, 7]);  as_strided_4 = None
    div_1: "f32[8, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_233: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, permute_151);  permute_151 = None
    mul_234: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(div_1, view_179);  view_179 = None
    sum_6: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 2, 3], True);  mul_233 = None
    view_183: "f32[1024]" = torch.ops.aten.reshape.default(sum_6, [1024]);  sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_161: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_234, [0, 2, 3, 1]);  mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_74: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_184: "f32[392, 1024]" = torch.ops.aten.reshape.default(clone_74, [392, 1024]);  clone_74 = None
    mm_2: "f32[392, 4096]" = torch.ops.aten.mm.default(view_184, permute_162);  permute_162 = None
    permute_163: "f32[1024, 392]" = torch.ops.aten.permute.default(view_184, [1, 0])
    mm_3: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_163, view_177);  permute_163 = view_177 = None
    permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True);  view_184 = None
    view_185: "f32[1024]" = torch.ops.aten.reshape.default(sum_7, [1024]);  sum_7 = None
    permute_165: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_186: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(mm_2, [8, 7, 7, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_236: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_150, 0.5);  add_150 = None
    mul_237: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_238: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_237, -0.5);  mul_237 = None
    exp: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_238);  mul_238 = None
    mul_239: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_240: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_176, mul_239);  view_176 = mul_239 = None
    add_155: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_236, mul_240);  mul_236 = mul_240 = None
    mul_241: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_186, add_155);  view_186 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_187: "f32[392, 4096]" = torch.ops.aten.reshape.default(mul_241, [392, 4096]);  mul_241 = None
    mm_4: "f32[392, 1024]" = torch.ops.aten.mm.default(view_187, permute_166);  permute_166 = None
    permute_167: "f32[4096, 392]" = torch.ops.aten.permute.default(view_187, [1, 0])
    mm_5: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_167, view_175);  permute_167 = view_175 = None
    permute_168: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_187, [0], True);  view_187 = None
    view_188: "f32[4096]" = torch.ops.aten.reshape.default(sum_8, [4096]);  sum_8 = None
    permute_169: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_189: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_4, [8, 7, 7, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_243: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_189, primals_114);  primals_114 = None
    mul_244: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_243, 1024)
    sum_9: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [3], True)
    mul_245: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_243, mul_218);  mul_243 = None
    sum_10: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [3], True);  mul_245 = None
    mul_246: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_218, sum_10);  sum_10 = None
    sub_45: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_244, sum_9);  mul_244 = sum_9 = None
    sub_46: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_45, mul_246);  sub_45 = mul_246 = None
    div_2: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None
    mul_247: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_46);  div_2 = sub_46 = None
    mul_248: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_189, mul_218);  mul_218 = None
    sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1, 2]);  mul_248 = None
    sum_12: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_189, [0, 1, 2]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_170: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_247, [0, 3, 1, 2]);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(permute_170, add_147, primals_337, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_170 = add_147 = primals_337 = None
    getitem_82: "f32[8, 1024, 7, 7]" = convolution_backward[0]
    getitem_83: "f32[1024, 1, 7, 7]" = convolution_backward[1]
    getitem_84: "f32[1024]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_156: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(div_1, getitem_82);  div_1 = getitem_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_249: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_156, permute_147);  permute_147 = None
    mul_250: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_156, view_174);  view_174 = None
    sum_13: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [0, 2, 3], True);  mul_249 = None
    view_190: "f32[1024]" = torch.ops.aten.reshape.default(sum_13, [1024]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_171: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_250, [0, 2, 3, 1]);  mul_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_75: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_191: "f32[392, 1024]" = torch.ops.aten.reshape.default(clone_75, [392, 1024]);  clone_75 = None
    mm_6: "f32[392, 4096]" = torch.ops.aten.mm.default(view_191, permute_172);  permute_172 = None
    permute_173: "f32[1024, 392]" = torch.ops.aten.permute.default(view_191, [1, 0])
    mm_7: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_173, view_172);  permute_173 = view_172 = None
    permute_174: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_14: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
    view_192: "f32[1024]" = torch.ops.aten.reshape.default(sum_14, [1024]);  sum_14 = None
    permute_175: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_193: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(mm_6, [8, 7, 7, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_252: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
    mul_253: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, view_171)
    mul_254: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_253, -0.5);  mul_253 = None
    exp_1: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_254);  mul_254 = None
    mul_255: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_256: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_171, mul_255);  view_171 = mul_255 = None
    add_158: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_252, mul_256);  mul_252 = mul_256 = None
    mul_257: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_193, add_158);  view_193 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_194: "f32[392, 4096]" = torch.ops.aten.reshape.default(mul_257, [392, 4096]);  mul_257 = None
    mm_8: "f32[392, 1024]" = torch.ops.aten.mm.default(view_194, permute_176);  permute_176 = None
    permute_177: "f32[4096, 392]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_9: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_177, view_170);  permute_177 = view_170 = None
    permute_178: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_15: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[4096]" = torch.ops.aten.reshape.default(sum_15, [4096]);  sum_15 = None
    permute_179: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_196: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_8, [8, 7, 7, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_259: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_196, primals_111);  primals_111 = None
    mul_260: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_259, 1024)
    sum_16: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [3], True)
    mul_261: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_259, mul_212);  mul_259 = None
    sum_17: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [3], True);  mul_261 = None
    mul_262: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_212, sum_17);  sum_17 = None
    sub_48: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_260, sum_16);  mul_260 = sum_16 = None
    sub_49: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_48, mul_262);  sub_48 = mul_262 = None
    div_3: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None
    mul_263: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_49);  div_3 = sub_49 = None
    mul_264: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_196, mul_212);  mul_212 = None
    sum_18: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1, 2]);  mul_264 = None
    sum_19: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_196, [0, 1, 2]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_180: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_263, [0, 3, 1, 2]);  mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(permute_180, add_143, primals_331, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_180 = add_143 = primals_331 = None
    getitem_85: "f32[8, 1024, 7, 7]" = convolution_backward_1[0]
    getitem_86: "f32[1024, 1, 7, 7]" = convolution_backward_1[1]
    getitem_87: "f32[1024]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_159: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_156, getitem_85);  add_156 = getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_265: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_159, permute_143);  permute_143 = None
    mul_266: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(add_159, view_169);  view_169 = None
    sum_20: "f32[1, 1024, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 2, 3], True);  mul_265 = None
    view_197: "f32[1024]" = torch.ops.aten.reshape.default(sum_20, [1024]);  sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_181: "f32[8, 7, 7, 1024]" = torch.ops.aten.permute.default(mul_266, [0, 2, 3, 1]);  mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_76: "f32[8, 7, 7, 1024]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_198: "f32[392, 1024]" = torch.ops.aten.reshape.default(clone_76, [392, 1024]);  clone_76 = None
    mm_10: "f32[392, 4096]" = torch.ops.aten.mm.default(view_198, permute_182);  permute_182 = None
    permute_183: "f32[1024, 392]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_11: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_183, view_167);  permute_183 = view_167 = None
    permute_184: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_21: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[1024]" = torch.ops.aten.reshape.default(sum_21, [1024]);  sum_21 = None
    permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_200: "f32[8, 7, 7, 4096]" = torch.ops.aten.reshape.default(mm_10, [8, 7, 7, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_268: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_269: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, view_166)
    mul_270: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(mul_269, -0.5);  mul_269 = None
    exp_2: "f32[8, 7, 7, 4096]" = torch.ops.aten.exp.default(mul_270);  mul_270 = None
    mul_271: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_272: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_166, mul_271);  view_166 = mul_271 = None
    add_161: "f32[8, 7, 7, 4096]" = torch.ops.aten.add.Tensor(mul_268, mul_272);  mul_268 = mul_272 = None
    mul_273: "f32[8, 7, 7, 4096]" = torch.ops.aten.mul.Tensor(view_200, add_161);  view_200 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_201: "f32[392, 4096]" = torch.ops.aten.reshape.default(mul_273, [392, 4096]);  mul_273 = None
    mm_12: "f32[392, 1024]" = torch.ops.aten.mm.default(view_201, permute_186);  permute_186 = None
    permute_187: "f32[4096, 392]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_13: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_187, view_165);  permute_187 = view_165 = None
    permute_188: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_22: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[4096]" = torch.ops.aten.reshape.default(sum_22, [4096]);  sum_22 = None
    permute_189: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    view_203: "f32[8, 7, 7, 1024]" = torch.ops.aten.reshape.default(mm_12, [8, 7, 7, 1024]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_275: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_203, primals_108);  primals_108 = None
    mul_276: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, 1024)
    sum_23: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [3], True)
    mul_277: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_275, mul_206);  mul_275 = None
    sum_24: "f32[8, 7, 7, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [3], True);  mul_277 = None
    mul_278: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(mul_206, sum_24);  sum_24 = None
    sub_51: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(mul_276, sum_23);  mul_276 = sum_23 = None
    sub_52: "f32[8, 7, 7, 1024]" = torch.ops.aten.sub.Tensor(sub_51, mul_278);  sub_51 = mul_278 = None
    div_4: "f32[8, 7, 7, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None
    mul_279: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_4, sub_52);  div_4 = sub_52 = None
    mul_280: "f32[8, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_203, mul_206);  mul_206 = None
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1, 2]);  mul_280 = None
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1, 2]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_190: "f32[8, 1024, 7, 7]" = torch.ops.aten.permute.default(mul_279, [0, 3, 1, 2]);  mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_190, convolution_36, primals_325, [1024], [1, 1], [3, 3], [1, 1], False, [0, 0], 1024, [True, True, True]);  permute_190 = convolution_36 = primals_325 = None
    getitem_88: "f32[8, 1024, 7, 7]" = convolution_backward_2[0]
    getitem_89: "f32[1024, 1, 7, 7]" = convolution_backward_2[1]
    getitem_90: "f32[1024]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_162: "f32[8, 1024, 7, 7]" = torch.ops.aten.add.Tensor(add_159, getitem_88);  add_159 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_162, permute_139, primals_323, [1024], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  add_162 = permute_139 = primals_323 = None
    getitem_91: "f32[8, 512, 14, 14]" = convolution_backward_3[0]
    getitem_92: "f32[1024, 512, 2, 2]" = convolution_backward_3[1]
    getitem_93: "f32[1024]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_191: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(getitem_91, [0, 2, 3, 1]);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_282: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(permute_191, primals_106);  primals_106 = None
    mul_283: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, 512)
    sum_27: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [3], True)
    mul_284: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_282, mul_204);  mul_282 = None
    sum_28: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [3], True);  mul_284 = None
    mul_285: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_204, sum_28);  sum_28 = None
    sub_54: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_283, sum_27);  mul_283 = sum_27 = None
    sub_55: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_54, mul_285);  sub_54 = mul_285 = None
    mul_286: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_5, sub_55);  div_5 = sub_55 = None
    mul_287: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(permute_191, mul_204);  mul_204 = None
    sum_29: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1, 2]);  mul_287 = None
    sum_30: "f32[512]" = torch.ops.aten.sum.dim_IntList(permute_191, [0, 1, 2]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_192: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_286, [0, 3, 1, 2]);  mul_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_288: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_192, permute_137);  permute_137 = None
    mul_289: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(permute_192, view_164);  view_164 = None
    sum_31: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 2, 3], True);  mul_288 = None
    view_204: "f32[512]" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_193: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_289, [0, 2, 3, 1]);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_205: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_193, [1568, 512]);  permute_193 = None
    mm_14: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_205, permute_194);  permute_194 = None
    permute_195: "f32[512, 1568]" = torch.ops.aten.permute.default(view_205, [1, 0])
    mm_15: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_195, view_162);  permute_195 = view_162 = None
    permute_196: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_32: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_205, [0], True);  view_205 = None
    view_206: "f32[512]" = torch.ops.aten.reshape.default(sum_32, [512]);  sum_32 = None
    permute_197: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_207: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_14, [8, 14, 14, 2048]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_291: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_292: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, view_161)
    mul_293: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_292, -0.5);  mul_292 = None
    exp_3: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_293);  mul_293 = None
    mul_294: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_295: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_161, mul_294);  view_161 = mul_294 = None
    add_164: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_291, mul_295);  mul_291 = mul_295 = None
    mul_296: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_207, add_164);  view_207 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_208: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_296, [1568, 2048]);  mul_296 = None
    mm_16: "f32[1568, 512]" = torch.ops.aten.mm.default(view_208, permute_198);  permute_198 = None
    permute_199: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_208, [1, 0])
    mm_17: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_199, view_160);  permute_199 = view_160 = None
    permute_200: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_33: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
    view_209: "f32[2048]" = torch.ops.aten.reshape.default(sum_33, [2048]);  sum_33 = None
    permute_201: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_210: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_16, [8, 14, 14, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_298: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_210, primals_103);  primals_103 = None
    mul_299: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_298, 512)
    sum_34: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [3], True)
    mul_300: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_298, mul_198);  mul_298 = None
    sum_35: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [3], True);  mul_300 = None
    mul_301: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_198, sum_35);  sum_35 = None
    sub_57: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_299, sum_34);  mul_299 = sum_34 = None
    sub_58: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_57, mul_301);  sub_57 = mul_301 = None
    div_6: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 512);  rsqrt_35 = None
    mul_302: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_6, sub_58);  div_6 = sub_58 = None
    mul_303: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_210, mul_198);  mul_198 = None
    sum_36: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1, 2]);  mul_303 = None
    sum_37: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_210, [0, 1, 2]);  view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_202: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_302, [0, 3, 1, 2]);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(permute_202, add_133, primals_317, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_202 = add_133 = primals_317 = None
    getitem_94: "f32[8, 512, 14, 14]" = convolution_backward_4[0]
    getitem_95: "f32[512, 1, 7, 7]" = convolution_backward_4[1]
    getitem_96: "f32[512]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_165: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(permute_192, getitem_94);  permute_192 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_304: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, permute_133);  permute_133 = None
    mul_305: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_165, view_159);  view_159 = None
    sum_38: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 2, 3], True);  mul_304 = None
    view_211: "f32[512]" = torch.ops.aten.reshape.default(sum_38, [512]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_203: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_305, [0, 2, 3, 1]);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_203, [1568, 512]);  permute_203 = None
    mm_18: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_212, permute_204);  permute_204 = None
    permute_205: "f32[512, 1568]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_19: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_205, view_157);  permute_205 = view_157 = None
    permute_206: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_39: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[512]" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None
    permute_207: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_214: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_18, [8, 14, 14, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_307: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_132, 0.5);  add_132 = None
    mul_308: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, view_156)
    mul_309: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_308, -0.5);  mul_308 = None
    exp_4: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_309);  mul_309 = None
    mul_310: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_311: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_156, mul_310);  view_156 = mul_310 = None
    add_167: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_307, mul_311);  mul_307 = mul_311 = None
    mul_312: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_214, add_167);  view_214 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_215: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_312, [1568, 2048]);  mul_312 = None
    mm_20: "f32[1568, 512]" = torch.ops.aten.mm.default(view_215, permute_208);  permute_208 = None
    permute_209: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_21: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_209, view_155);  permute_209 = view_155 = None
    permute_210: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_40: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[2048]" = torch.ops.aten.reshape.default(sum_40, [2048]);  sum_40 = None
    permute_211: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_217: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_20, [8, 14, 14, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_314: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_217, primals_100);  primals_100 = None
    mul_315: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_314, 512)
    sum_41: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [3], True)
    mul_316: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_314, mul_192);  mul_314 = None
    sum_42: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [3], True);  mul_316 = None
    mul_317: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_192, sum_42);  sum_42 = None
    sub_60: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_315, sum_41);  mul_315 = sum_41 = None
    sub_61: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_60, mul_317);  sub_60 = mul_317 = None
    div_7: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 512);  rsqrt_34 = None
    mul_318: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_7, sub_61);  div_7 = sub_61 = None
    mul_319: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_217, mul_192);  mul_192 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1, 2]);  mul_319 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1, 2]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_212: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_318, [0, 3, 1, 2]);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(permute_212, add_129, primals_311, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_212 = add_129 = primals_311 = None
    getitem_97: "f32[8, 512, 14, 14]" = convolution_backward_5[0]
    getitem_98: "f32[512, 1, 7, 7]" = convolution_backward_5[1]
    getitem_99: "f32[512]" = convolution_backward_5[2];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_168: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_165, getitem_97);  add_165 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_320: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_168, permute_129);  permute_129 = None
    mul_321: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_168, view_154);  view_154 = None
    sum_45: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3], True);  mul_320 = None
    view_218: "f32[512]" = torch.ops.aten.reshape.default(sum_45, [512]);  sum_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_213: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_321, [0, 2, 3, 1]);  mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_213, [1568, 512]);  permute_213 = None
    mm_22: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_219, permute_214);  permute_214 = None
    permute_215: "f32[512, 1568]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_23: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_215, view_152);  permute_215 = view_152 = None
    permute_216: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_46: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[512]" = torch.ops.aten.reshape.default(sum_46, [512]);  sum_46 = None
    permute_217: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_221: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_22, [8, 14, 14, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_323: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_324: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_325: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_324, -0.5);  mul_324 = None
    exp_5: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_325);  mul_325 = None
    mul_326: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_327: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_151, mul_326);  view_151 = mul_326 = None
    add_170: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_323, mul_327);  mul_323 = mul_327 = None
    mul_328: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_221, add_170);  view_221 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_222: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_328, [1568, 2048]);  mul_328 = None
    mm_24: "f32[1568, 512]" = torch.ops.aten.mm.default(view_222, permute_218);  permute_218 = None
    permute_219: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_25: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_219, view_150);  permute_219 = view_150 = None
    permute_220: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_47: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[2048]" = torch.ops.aten.reshape.default(sum_47, [2048]);  sum_47 = None
    permute_221: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_224: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_24, [8, 14, 14, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_330: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_224, primals_97);  primals_97 = None
    mul_331: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_330, 512)
    sum_48: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [3], True)
    mul_332: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_330, mul_186);  mul_330 = None
    sum_49: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [3], True);  mul_332 = None
    mul_333: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_186, sum_49);  sum_49 = None
    sub_63: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_331, sum_48);  mul_331 = sum_48 = None
    sub_64: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_63, mul_333);  sub_63 = mul_333 = None
    div_8: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 512);  rsqrt_33 = None
    mul_334: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_64);  div_8 = sub_64 = None
    mul_335: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_224, mul_186);  mul_186 = None
    sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1, 2]);  mul_335 = None
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1, 2]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_222: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_334, [0, 3, 1, 2]);  mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(permute_222, add_125, primals_305, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_222 = add_125 = primals_305 = None
    getitem_100: "f32[8, 512, 14, 14]" = convolution_backward_6[0]
    getitem_101: "f32[512, 1, 7, 7]" = convolution_backward_6[1]
    getitem_102: "f32[512]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_171: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_168, getitem_100);  add_168 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_336: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_171, permute_125);  permute_125 = None
    mul_337: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_171, view_149);  view_149 = None
    sum_52: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3], True);  mul_336 = None
    view_225: "f32[512]" = torch.ops.aten.reshape.default(sum_52, [512]);  sum_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_223: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_337, [0, 2, 3, 1]);  mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_226: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_223, [1568, 512]);  permute_223 = None
    mm_26: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_226, permute_224);  permute_224 = None
    permute_225: "f32[512, 1568]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_27: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_225, view_147);  permute_225 = view_147 = None
    permute_226: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_53: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[512]" = torch.ops.aten.reshape.default(sum_53, [512]);  sum_53 = None
    permute_227: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_228: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_26, [8, 14, 14, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_339: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_124, 0.5);  add_124 = None
    mul_340: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_341: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
    exp_6: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
    mul_342: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_343: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_146, mul_342);  view_146 = mul_342 = None
    add_173: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
    mul_344: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_228, add_173);  view_228 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_229: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_344, [1568, 2048]);  mul_344 = None
    mm_28: "f32[1568, 512]" = torch.ops.aten.mm.default(view_229, permute_228);  permute_228 = None
    permute_229: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_229, [1, 0])
    mm_29: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_229, view_145);  permute_229 = view_145 = None
    permute_230: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_54: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_229, [0], True);  view_229 = None
    view_230: "f32[2048]" = torch.ops.aten.reshape.default(sum_54, [2048]);  sum_54 = None
    permute_231: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_231: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_28, [8, 14, 14, 512]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_346: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_231, primals_94);  primals_94 = None
    mul_347: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_346, 512)
    sum_55: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [3], True)
    mul_348: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_346, mul_180);  mul_346 = None
    sum_56: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [3], True);  mul_348 = None
    mul_349: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_180, sum_56);  sum_56 = None
    sub_66: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_347, sum_55);  mul_347 = sum_55 = None
    sub_67: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_66, mul_349);  sub_66 = mul_349 = None
    div_9: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 512);  rsqrt_32 = None
    mul_350: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_9, sub_67);  div_9 = sub_67 = None
    mul_351: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_231, mul_180);  mul_180 = None
    sum_57: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1, 2]);  mul_351 = None
    sum_58: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_231, [0, 1, 2]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_232: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_350, [0, 3, 1, 2]);  mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(permute_232, add_121, primals_299, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_232 = add_121 = primals_299 = None
    getitem_103: "f32[8, 512, 14, 14]" = convolution_backward_7[0]
    getitem_104: "f32[512, 1, 7, 7]" = convolution_backward_7[1]
    getitem_105: "f32[512]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_174: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_171, getitem_103);  add_171 = getitem_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_352: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, permute_121);  permute_121 = None
    mul_353: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_174, view_144);  view_144 = None
    sum_59: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3], True);  mul_352 = None
    view_232: "f32[512]" = torch.ops.aten.reshape.default(sum_59, [512]);  sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_233: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_353, [0, 2, 3, 1]);  mul_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_233: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_233, [1568, 512]);  permute_233 = None
    mm_30: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_233, permute_234);  permute_234 = None
    permute_235: "f32[512, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_31: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_235, view_142);  permute_235 = view_142 = None
    permute_236: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_60: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[512]" = torch.ops.aten.reshape.default(sum_60, [512]);  sum_60 = None
    permute_237: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_235: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_30, [8, 14, 14, 2048]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_355: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
    mul_356: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_357: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_356, -0.5);  mul_356 = None
    exp_7: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_357);  mul_357 = None
    mul_358: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_359: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_141, mul_358);  view_141 = mul_358 = None
    add_176: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_355, mul_359);  mul_355 = mul_359 = None
    mul_360: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_235, add_176);  view_235 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_236: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_360, [1568, 2048]);  mul_360 = None
    mm_32: "f32[1568, 512]" = torch.ops.aten.mm.default(view_236, permute_238);  permute_238 = None
    permute_239: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_33: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_239, view_140);  permute_239 = view_140 = None
    permute_240: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_61: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[2048]" = torch.ops.aten.reshape.default(sum_61, [2048]);  sum_61 = None
    permute_241: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_238: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_32, [8, 14, 14, 512]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_362: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_238, primals_91);  primals_91 = None
    mul_363: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_362, 512)
    sum_62: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [3], True)
    mul_364: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_362, mul_174);  mul_362 = None
    sum_63: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [3], True);  mul_364 = None
    mul_365: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_174, sum_63);  sum_63 = None
    sub_69: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_363, sum_62);  mul_363 = sum_62 = None
    sub_70: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_69, mul_365);  sub_69 = mul_365 = None
    div_10: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 512);  rsqrt_31 = None
    mul_366: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_10, sub_70);  div_10 = sub_70 = None
    mul_367: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_238, mul_174);  mul_174 = None
    sum_64: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_367, [0, 1, 2]);  mul_367 = None
    sum_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_238, [0, 1, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_242: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_366, [0, 3, 1, 2]);  mul_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(permute_242, add_117, primals_293, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_242 = add_117 = primals_293 = None
    getitem_106: "f32[8, 512, 14, 14]" = convolution_backward_8[0]
    getitem_107: "f32[512, 1, 7, 7]" = convolution_backward_8[1]
    getitem_108: "f32[512]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_177: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_174, getitem_106);  add_174 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_368: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, permute_117);  permute_117 = None
    mul_369: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_177, view_139);  view_139 = None
    sum_66: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3], True);  mul_368 = None
    view_239: "f32[512]" = torch.ops.aten.reshape.default(sum_66, [512]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_243: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_369, [0, 2, 3, 1]);  mul_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_240: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_243, [1568, 512]);  permute_243 = None
    mm_34: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_240, permute_244);  permute_244 = None
    permute_245: "f32[512, 1568]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_35: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_245, view_137);  permute_245 = view_137 = None
    permute_246: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_67: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
    view_241: "f32[512]" = torch.ops.aten.reshape.default(sum_67, [512]);  sum_67 = None
    permute_247: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_242: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_34, [8, 14, 14, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_371: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
    mul_372: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, view_136)
    mul_373: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_372, -0.5);  mul_372 = None
    exp_8: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_373);  mul_373 = None
    mul_374: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_375: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_136, mul_374);  view_136 = mul_374 = None
    add_179: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_371, mul_375);  mul_371 = mul_375 = None
    mul_376: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_242, add_179);  view_242 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_243: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_376, [1568, 2048]);  mul_376 = None
    mm_36: "f32[1568, 512]" = torch.ops.aten.mm.default(view_243, permute_248);  permute_248 = None
    permute_249: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_37: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_249, view_135);  permute_249 = view_135 = None
    permute_250: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_68: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[2048]" = torch.ops.aten.reshape.default(sum_68, [2048]);  sum_68 = None
    permute_251: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_245: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_36, [8, 14, 14, 512]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_378: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_245, primals_88);  primals_88 = None
    mul_379: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_378, 512)
    sum_69: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_378, [3], True)
    mul_380: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_378, mul_168);  mul_378 = None
    sum_70: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [3], True);  mul_380 = None
    mul_381: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_168, sum_70);  sum_70 = None
    sub_72: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_379, sum_69);  mul_379 = sum_69 = None
    sub_73: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_72, mul_381);  sub_72 = mul_381 = None
    div_11: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 512);  rsqrt_30 = None
    mul_382: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_73);  div_11 = sub_73 = None
    mul_383: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_245, mul_168);  mul_168 = None
    sum_71: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 1, 2]);  mul_383 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1, 2]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_252: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_382, [0, 3, 1, 2]);  mul_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(permute_252, add_113, primals_287, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_252 = add_113 = primals_287 = None
    getitem_109: "f32[8, 512, 14, 14]" = convolution_backward_9[0]
    getitem_110: "f32[512, 1, 7, 7]" = convolution_backward_9[1]
    getitem_111: "f32[512]" = convolution_backward_9[2];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_180: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_177, getitem_109);  add_177 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_384: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, permute_113);  permute_113 = None
    mul_385: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_180, view_134);  view_134 = None
    sum_73: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3], True);  mul_384 = None
    view_246: "f32[512]" = torch.ops.aten.reshape.default(sum_73, [512]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_253: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_385, [0, 2, 3, 1]);  mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_247: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_253, [1568, 512]);  permute_253 = None
    mm_38: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_247, permute_254);  permute_254 = None
    permute_255: "f32[512, 1568]" = torch.ops.aten.permute.default(view_247, [1, 0])
    mm_39: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_255, view_132);  permute_255 = view_132 = None
    permute_256: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_74: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True);  view_247 = None
    view_248: "f32[512]" = torch.ops.aten.reshape.default(sum_74, [512]);  sum_74 = None
    permute_257: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_249: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_38, [8, 14, 14, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_387: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
    mul_388: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_389: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_388, -0.5);  mul_388 = None
    exp_9: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_389);  mul_389 = None
    mul_390: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_391: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_131, mul_390);  view_131 = mul_390 = None
    add_182: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_387, mul_391);  mul_387 = mul_391 = None
    mul_392: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_249, add_182);  view_249 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_250: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_392, [1568, 2048]);  mul_392 = None
    mm_40: "f32[1568, 512]" = torch.ops.aten.mm.default(view_250, permute_258);  permute_258 = None
    permute_259: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_250, [1, 0])
    mm_41: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_259, view_130);  permute_259 = view_130 = None
    permute_260: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_75: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
    view_251: "f32[2048]" = torch.ops.aten.reshape.default(sum_75, [2048]);  sum_75 = None
    permute_261: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_252: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_40, [8, 14, 14, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_394: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_252, primals_85);  primals_85 = None
    mul_395: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_394, 512)
    sum_76: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_394, [3], True)
    mul_396: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_394, mul_162);  mul_394 = None
    sum_77: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [3], True);  mul_396 = None
    mul_397: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_162, sum_77);  sum_77 = None
    sub_75: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_395, sum_76);  mul_395 = sum_76 = None
    sub_76: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_75, mul_397);  sub_75 = mul_397 = None
    div_12: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 512);  rsqrt_29 = None
    mul_398: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_12, sub_76);  div_12 = sub_76 = None
    mul_399: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_252, mul_162);  mul_162 = None
    sum_78: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 1, 2]);  mul_399 = None
    sum_79: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_252, [0, 1, 2]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_262: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_398, [0, 3, 1, 2]);  mul_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(permute_262, add_109, primals_281, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_262 = add_109 = primals_281 = None
    getitem_112: "f32[8, 512, 14, 14]" = convolution_backward_10[0]
    getitem_113: "f32[512, 1, 7, 7]" = convolution_backward_10[1]
    getitem_114: "f32[512]" = convolution_backward_10[2];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_183: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_180, getitem_112);  add_180 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_400: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_183, permute_109);  permute_109 = None
    mul_401: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_183, view_129);  view_129 = None
    sum_80: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3], True);  mul_400 = None
    view_253: "f32[512]" = torch.ops.aten.reshape.default(sum_80, [512]);  sum_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_263: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_401, [0, 2, 3, 1]);  mul_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_254: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_263, [1568, 512]);  permute_263 = None
    mm_42: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_254, permute_264);  permute_264 = None
    permute_265: "f32[512, 1568]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_43: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_265, view_127);  permute_265 = view_127 = None
    permute_266: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_81: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[512]" = torch.ops.aten.reshape.default(sum_81, [512]);  sum_81 = None
    permute_267: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_256: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_42, [8, 14, 14, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_403: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_404: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, view_126)
    mul_405: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_404, -0.5);  mul_404 = None
    exp_10: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_405);  mul_405 = None
    mul_406: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_407: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_126, mul_406);  view_126 = mul_406 = None
    add_185: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_403, mul_407);  mul_403 = mul_407 = None
    mul_408: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_256, add_185);  view_256 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_257: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_408, [1568, 2048]);  mul_408 = None
    mm_44: "f32[1568, 512]" = torch.ops.aten.mm.default(view_257, permute_268);  permute_268 = None
    permute_269: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_45: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_269, view_125);  permute_269 = view_125 = None
    permute_270: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_82: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[2048]" = torch.ops.aten.reshape.default(sum_82, [2048]);  sum_82 = None
    permute_271: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_259: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_44, [8, 14, 14, 512]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_410: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_259, primals_82);  primals_82 = None
    mul_411: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_410, 512)
    sum_83: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_410, [3], True)
    mul_412: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_410, mul_156);  mul_410 = None
    sum_84: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [3], True);  mul_412 = None
    mul_413: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_156, sum_84);  sum_84 = None
    sub_78: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_411, sum_83);  mul_411 = sum_83 = None
    sub_79: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_78, mul_413);  sub_78 = mul_413 = None
    div_13: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 512);  rsqrt_28 = None
    mul_414: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_13, sub_79);  div_13 = sub_79 = None
    mul_415: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_259, mul_156);  mul_156 = None
    sum_85: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 1, 2]);  mul_415 = None
    sum_86: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_259, [0, 1, 2]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_272: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_414, [0, 3, 1, 2]);  mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(permute_272, add_105, primals_275, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_272 = add_105 = primals_275 = None
    getitem_115: "f32[8, 512, 14, 14]" = convolution_backward_11[0]
    getitem_116: "f32[512, 1, 7, 7]" = convolution_backward_11[1]
    getitem_117: "f32[512]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_186: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_183, getitem_115);  add_183 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_416: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_186, permute_105);  permute_105 = None
    mul_417: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_186, view_124);  view_124 = None
    sum_87: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 2, 3], True);  mul_416 = None
    view_260: "f32[512]" = torch.ops.aten.reshape.default(sum_87, [512]);  sum_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_273: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_417, [0, 2, 3, 1]);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_261: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_273, [1568, 512]);  permute_273 = None
    mm_46: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_261, permute_274);  permute_274 = None
    permute_275: "f32[512, 1568]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_47: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_275, view_122);  permute_275 = view_122 = None
    permute_276: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_88: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[512]" = torch.ops.aten.reshape.default(sum_88, [512]);  sum_88 = None
    permute_277: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_263: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_46, [8, 14, 14, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_419: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
    mul_420: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, view_121)
    mul_421: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_420, -0.5);  mul_420 = None
    exp_11: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_421);  mul_421 = None
    mul_422: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_423: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_121, mul_422);  view_121 = mul_422 = None
    add_188: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_419, mul_423);  mul_419 = mul_423 = None
    mul_424: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_263, add_188);  view_263 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_264: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_424, [1568, 2048]);  mul_424 = None
    mm_48: "f32[1568, 512]" = torch.ops.aten.mm.default(view_264, permute_278);  permute_278 = None
    permute_279: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_49: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_279, view_120);  permute_279 = view_120 = None
    permute_280: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_89: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[2048]" = torch.ops.aten.reshape.default(sum_89, [2048]);  sum_89 = None
    permute_281: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_266: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_48, [8, 14, 14, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_426: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_266, primals_79);  primals_79 = None
    mul_427: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_426, 512)
    sum_90: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [3], True)
    mul_428: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_426, mul_150);  mul_426 = None
    sum_91: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [3], True);  mul_428 = None
    mul_429: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_150, sum_91);  sum_91 = None
    sub_81: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_427, sum_90);  mul_427 = sum_90 = None
    sub_82: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_81, mul_429);  sub_81 = mul_429 = None
    div_14: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 512);  rsqrt_27 = None
    mul_430: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_14, sub_82);  div_14 = sub_82 = None
    mul_431: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_266, mul_150);  mul_150 = None
    sum_92: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1, 2]);  mul_431 = None
    sum_93: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_266, [0, 1, 2]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_282: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_430, [0, 3, 1, 2]);  mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_282, add_101, primals_269, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_282 = add_101 = primals_269 = None
    getitem_118: "f32[8, 512, 14, 14]" = convolution_backward_12[0]
    getitem_119: "f32[512, 1, 7, 7]" = convolution_backward_12[1]
    getitem_120: "f32[512]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_189: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_186, getitem_118);  add_186 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_432: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, permute_101);  permute_101 = None
    mul_433: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, view_119);  view_119 = None
    sum_94: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 2, 3], True);  mul_432 = None
    view_267: "f32[512]" = torch.ops.aten.reshape.default(sum_94, [512]);  sum_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_283: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_433, [0, 2, 3, 1]);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_268: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_283, [1568, 512]);  permute_283 = None
    mm_50: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_268, permute_284);  permute_284 = None
    permute_285: "f32[512, 1568]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_51: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_285, view_117);  permute_285 = view_117 = None
    permute_286: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_95: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[512]" = torch.ops.aten.reshape.default(sum_95, [512]);  sum_95 = None
    permute_287: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_270: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_50, [8, 14, 14, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_435: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_100, 0.5);  add_100 = None
    mul_436: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, view_116)
    mul_437: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_436, -0.5);  mul_436 = None
    exp_12: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_437);  mul_437 = None
    mul_438: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_439: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_116, mul_438);  view_116 = mul_438 = None
    add_191: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_435, mul_439);  mul_435 = mul_439 = None
    mul_440: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_270, add_191);  view_270 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_271: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_440, [1568, 2048]);  mul_440 = None
    mm_52: "f32[1568, 512]" = torch.ops.aten.mm.default(view_271, permute_288);  permute_288 = None
    permute_289: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_53: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_289, view_115);  permute_289 = view_115 = None
    permute_290: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_96: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[2048]" = torch.ops.aten.reshape.default(sum_96, [2048]);  sum_96 = None
    permute_291: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_273: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_52, [8, 14, 14, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_442: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_273, primals_76);  primals_76 = None
    mul_443: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_442, 512)
    sum_97: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [3], True)
    mul_444: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_442, mul_144);  mul_442 = None
    sum_98: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [3], True);  mul_444 = None
    mul_445: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_144, sum_98);  sum_98 = None
    sub_84: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_443, sum_97);  mul_443 = sum_97 = None
    sub_85: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_84, mul_445);  sub_84 = mul_445 = None
    div_15: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
    mul_446: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_15, sub_85);  div_15 = sub_85 = None
    mul_447: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_273, mul_144);  mul_144 = None
    sum_99: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1, 2]);  mul_447 = None
    sum_100: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1, 2]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_292: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_446, [0, 3, 1, 2]);  mul_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(permute_292, add_97, primals_263, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_292 = add_97 = primals_263 = None
    getitem_121: "f32[8, 512, 14, 14]" = convolution_backward_13[0]
    getitem_122: "f32[512, 1, 7, 7]" = convolution_backward_13[1]
    getitem_123: "f32[512]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_192: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_189, getitem_121);  add_189 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_448: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, permute_97);  permute_97 = None
    mul_449: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_192, view_114);  view_114 = None
    sum_101: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3], True);  mul_448 = None
    view_274: "f32[512]" = torch.ops.aten.reshape.default(sum_101, [512]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_293: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_449, [0, 2, 3, 1]);  mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_275: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_293, [1568, 512]);  permute_293 = None
    mm_54: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_275, permute_294);  permute_294 = None
    permute_295: "f32[512, 1568]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_55: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_295, view_112);  permute_295 = view_112 = None
    permute_296: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_102: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[512]" = torch.ops.aten.reshape.default(sum_102, [512]);  sum_102 = None
    permute_297: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_277: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_54, [8, 14, 14, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_451: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_452: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, view_111)
    mul_453: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_452, -0.5);  mul_452 = None
    exp_13: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_453);  mul_453 = None
    mul_454: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_455: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_111, mul_454);  view_111 = mul_454 = None
    add_194: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_451, mul_455);  mul_451 = mul_455 = None
    mul_456: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_277, add_194);  view_277 = add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_278: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_456, [1568, 2048]);  mul_456 = None
    mm_56: "f32[1568, 512]" = torch.ops.aten.mm.default(view_278, permute_298);  permute_298 = None
    permute_299: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_57: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_299, view_110);  permute_299 = view_110 = None
    permute_300: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_103: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[2048]" = torch.ops.aten.reshape.default(sum_103, [2048]);  sum_103 = None
    permute_301: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_280: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_56, [8, 14, 14, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_458: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_280, primals_73);  primals_73 = None
    mul_459: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_458, 512)
    sum_104: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_458, [3], True)
    mul_460: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_458, mul_138);  mul_458 = None
    sum_105: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [3], True);  mul_460 = None
    mul_461: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_138, sum_105);  sum_105 = None
    sub_87: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_459, sum_104);  mul_459 = sum_104 = None
    sub_88: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_87, mul_461);  sub_87 = mul_461 = None
    div_16: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 512);  rsqrt_25 = None
    mul_462: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_16, sub_88);  div_16 = sub_88 = None
    mul_463: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_280, mul_138);  mul_138 = None
    sum_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 1, 2]);  mul_463 = None
    sum_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1, 2]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_302: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_462, [0, 3, 1, 2]);  mul_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(permute_302, add_93, primals_257, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_302 = add_93 = primals_257 = None
    getitem_124: "f32[8, 512, 14, 14]" = convolution_backward_14[0]
    getitem_125: "f32[512, 1, 7, 7]" = convolution_backward_14[1]
    getitem_126: "f32[512]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_195: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_192, getitem_124);  add_192 = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_464: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, permute_93);  permute_93 = None
    mul_465: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, view_109);  view_109 = None
    sum_108: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3], True);  mul_464 = None
    view_281: "f32[512]" = torch.ops.aten.reshape.default(sum_108, [512]);  sum_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_303: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_465, [0, 2, 3, 1]);  mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_282: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_303, [1568, 512]);  permute_303 = None
    mm_58: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_282, permute_304);  permute_304 = None
    permute_305: "f32[512, 1568]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_59: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_305, view_107);  permute_305 = view_107 = None
    permute_306: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_109: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[512]" = torch.ops.aten.reshape.default(sum_109, [512]);  sum_109 = None
    permute_307: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_284: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_58, [8, 14, 14, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_467: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_92, 0.5);  add_92 = None
    mul_468: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_469: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_468, -0.5);  mul_468 = None
    exp_14: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_469);  mul_469 = None
    mul_470: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_471: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_106, mul_470);  view_106 = mul_470 = None
    add_197: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_467, mul_471);  mul_467 = mul_471 = None
    mul_472: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_284, add_197);  view_284 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_285: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_472, [1568, 2048]);  mul_472 = None
    mm_60: "f32[1568, 512]" = torch.ops.aten.mm.default(view_285, permute_308);  permute_308 = None
    permute_309: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_61: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_309, view_105);  permute_309 = view_105 = None
    permute_310: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_110: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[2048]" = torch.ops.aten.reshape.default(sum_110, [2048]);  sum_110 = None
    permute_311: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_287: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_60, [8, 14, 14, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_474: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_287, primals_70);  primals_70 = None
    mul_475: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_474, 512)
    sum_111: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [3], True)
    mul_476: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_474, mul_132);  mul_474 = None
    sum_112: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [3], True);  mul_476 = None
    mul_477: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_132, sum_112);  sum_112 = None
    sub_90: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_475, sum_111);  mul_475 = sum_111 = None
    sub_91: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_90, mul_477);  sub_90 = mul_477 = None
    div_17: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 512);  rsqrt_24 = None
    mul_478: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_17, sub_91);  div_17 = sub_91 = None
    mul_479: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_287, mul_132);  mul_132 = None
    sum_113: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1, 2]);  mul_479 = None
    sum_114: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1, 2]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_312: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_478, [0, 3, 1, 2]);  mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(permute_312, add_89, primals_251, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_312 = add_89 = primals_251 = None
    getitem_127: "f32[8, 512, 14, 14]" = convolution_backward_15[0]
    getitem_128: "f32[512, 1, 7, 7]" = convolution_backward_15[1]
    getitem_129: "f32[512]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_198: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_195, getitem_127);  add_195 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_480: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, permute_89);  permute_89 = None
    mul_481: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_198, view_104);  view_104 = None
    sum_115: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3], True);  mul_480 = None
    view_288: "f32[512]" = torch.ops.aten.reshape.default(sum_115, [512]);  sum_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_313: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_481, [0, 2, 3, 1]);  mul_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_289: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_313, [1568, 512]);  permute_313 = None
    mm_62: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_289, permute_314);  permute_314 = None
    permute_315: "f32[512, 1568]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_63: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_315, view_102);  permute_315 = view_102 = None
    permute_316: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_116: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[512]" = torch.ops.aten.reshape.default(sum_116, [512]);  sum_116 = None
    permute_317: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_291: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_62, [8, 14, 14, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_483: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_484: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, view_101)
    mul_485: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_484, -0.5);  mul_484 = None
    exp_15: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_485);  mul_485 = None
    mul_486: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_487: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_101, mul_486);  view_101 = mul_486 = None
    add_200: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_483, mul_487);  mul_483 = mul_487 = None
    mul_488: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_291, add_200);  view_291 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_292: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_488, [1568, 2048]);  mul_488 = None
    mm_64: "f32[1568, 512]" = torch.ops.aten.mm.default(view_292, permute_318);  permute_318 = None
    permute_319: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_292, [1, 0])
    mm_65: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_319, view_100);  permute_319 = view_100 = None
    permute_320: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_117: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
    view_293: "f32[2048]" = torch.ops.aten.reshape.default(sum_117, [2048]);  sum_117 = None
    permute_321: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_294: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_64, [8, 14, 14, 512]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_490: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_294, primals_67);  primals_67 = None
    mul_491: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, 512)
    sum_118: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [3], True)
    mul_492: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_490, mul_126);  mul_490 = None
    sum_119: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [3], True);  mul_492 = None
    mul_493: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_126, sum_119);  sum_119 = None
    sub_93: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_491, sum_118);  mul_491 = sum_118 = None
    sub_94: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_93, mul_493);  sub_93 = mul_493 = None
    div_18: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 512);  rsqrt_23 = None
    mul_494: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_18, sub_94);  div_18 = sub_94 = None
    mul_495: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_294, mul_126);  mul_126 = None
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_495, [0, 1, 2]);  mul_495 = None
    sum_121: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_294, [0, 1, 2]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_322: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_494, [0, 3, 1, 2]);  mul_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(permute_322, add_85, primals_245, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_322 = add_85 = primals_245 = None
    getitem_130: "f32[8, 512, 14, 14]" = convolution_backward_16[0]
    getitem_131: "f32[512, 1, 7, 7]" = convolution_backward_16[1]
    getitem_132: "f32[512]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_201: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_198, getitem_130);  add_198 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_496: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_201, permute_85);  permute_85 = None
    mul_497: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_201, view_99);  view_99 = None
    sum_122: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 2, 3], True);  mul_496 = None
    view_295: "f32[512]" = torch.ops.aten.reshape.default(sum_122, [512]);  sum_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_323: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_497, [0, 2, 3, 1]);  mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_296: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_323, [1568, 512]);  permute_323 = None
    mm_66: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_296, permute_324);  permute_324 = None
    permute_325: "f32[512, 1568]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_67: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_325, view_97);  permute_325 = view_97 = None
    permute_326: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_123: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[512]" = torch.ops.aten.reshape.default(sum_123, [512]);  sum_123 = None
    permute_327: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_298: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_66, [8, 14, 14, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_499: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_500: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, view_96)
    mul_501: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_500, -0.5);  mul_500 = None
    exp_16: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_501);  mul_501 = None
    mul_502: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_503: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_96, mul_502);  view_96 = mul_502 = None
    add_203: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_499, mul_503);  mul_499 = mul_503 = None
    mul_504: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_298, add_203);  view_298 = add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_299: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_504, [1568, 2048]);  mul_504 = None
    mm_68: "f32[1568, 512]" = torch.ops.aten.mm.default(view_299, permute_328);  permute_328 = None
    permute_329: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_69: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_329, view_95);  permute_329 = view_95 = None
    permute_330: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_124: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[2048]" = torch.ops.aten.reshape.default(sum_124, [2048]);  sum_124 = None
    permute_331: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_301: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_68, [8, 14, 14, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_506: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_301, primals_64);  primals_64 = None
    mul_507: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, 512)
    sum_125: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [3], True)
    mul_508: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_506, mul_120);  mul_506 = None
    sum_126: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [3], True);  mul_508 = None
    mul_509: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_120, sum_126);  sum_126 = None
    sub_96: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_507, sum_125);  mul_507 = sum_125 = None
    sub_97: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_96, mul_509);  sub_96 = mul_509 = None
    div_19: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 512);  rsqrt_22 = None
    mul_510: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_19, sub_97);  div_19 = sub_97 = None
    mul_511: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_301, mul_120);  mul_120 = None
    sum_127: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1, 2]);  mul_511 = None
    sum_128: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1, 2]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_332: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_510, [0, 3, 1, 2]);  mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(permute_332, add_81, primals_239, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_332 = add_81 = primals_239 = None
    getitem_133: "f32[8, 512, 14, 14]" = convolution_backward_17[0]
    getitem_134: "f32[512, 1, 7, 7]" = convolution_backward_17[1]
    getitem_135: "f32[512]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_204: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_201, getitem_133);  add_201 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_512: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_204, permute_81);  permute_81 = None
    mul_513: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_204, view_94);  view_94 = None
    sum_129: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3], True);  mul_512 = None
    view_302: "f32[512]" = torch.ops.aten.reshape.default(sum_129, [512]);  sum_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_333: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_513, [0, 2, 3, 1]);  mul_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_303: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_333, [1568, 512]);  permute_333 = None
    mm_70: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_303, permute_334);  permute_334 = None
    permute_335: "f32[512, 1568]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_71: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_335, view_92);  permute_335 = view_92 = None
    permute_336: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_130: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_303, [0], True);  view_303 = None
    view_304: "f32[512]" = torch.ops.aten.reshape.default(sum_130, [512]);  sum_130 = None
    permute_337: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_305: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_70, [8, 14, 14, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_515: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_516: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, view_91)
    mul_517: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_516, -0.5);  mul_516 = None
    exp_17: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_517);  mul_517 = None
    mul_518: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_519: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_91, mul_518);  view_91 = mul_518 = None
    add_206: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_515, mul_519);  mul_515 = mul_519 = None
    mul_520: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_305, add_206);  view_305 = add_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_306: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_520, [1568, 2048]);  mul_520 = None
    mm_72: "f32[1568, 512]" = torch.ops.aten.mm.default(view_306, permute_338);  permute_338 = None
    permute_339: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_306, [1, 0])
    mm_73: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_339, view_90);  permute_339 = view_90 = None
    permute_340: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_131: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
    view_307: "f32[2048]" = torch.ops.aten.reshape.default(sum_131, [2048]);  sum_131 = None
    permute_341: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_308: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_72, [8, 14, 14, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_522: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_308, primals_61);  primals_61 = None
    mul_523: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_522, 512)
    sum_132: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [3], True)
    mul_524: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_522, mul_114);  mul_522 = None
    sum_133: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [3], True);  mul_524 = None
    mul_525: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_114, sum_133);  sum_133 = None
    sub_99: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_523, sum_132);  mul_523 = sum_132 = None
    sub_100: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_99, mul_525);  sub_99 = mul_525 = None
    div_20: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 512);  rsqrt_21 = None
    mul_526: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_20, sub_100);  div_20 = sub_100 = None
    mul_527: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_308, mul_114);  mul_114 = None
    sum_134: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 1, 2]);  mul_527 = None
    sum_135: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_308, [0, 1, 2]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_342: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_526, [0, 3, 1, 2]);  mul_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_342, add_77, primals_233, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_342 = add_77 = primals_233 = None
    getitem_136: "f32[8, 512, 14, 14]" = convolution_backward_18[0]
    getitem_137: "f32[512, 1, 7, 7]" = convolution_backward_18[1]
    getitem_138: "f32[512]" = convolution_backward_18[2];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_207: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_204, getitem_136);  add_204 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_528: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, permute_77);  permute_77 = None
    mul_529: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_207, view_89);  view_89 = None
    sum_136: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3], True);  mul_528 = None
    view_309: "f32[512]" = torch.ops.aten.reshape.default(sum_136, [512]);  sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_343: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_529, [0, 2, 3, 1]);  mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_343, [1568, 512]);  permute_343 = None
    mm_74: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_310, permute_344);  permute_344 = None
    permute_345: "f32[512, 1568]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_75: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_345, view_87);  permute_345 = view_87 = None
    permute_346: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_137: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[512]" = torch.ops.aten.reshape.default(sum_137, [512]);  sum_137 = None
    permute_347: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_312: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_74, [8, 14, 14, 2048]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_531: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
    mul_532: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, view_86)
    mul_533: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_532, -0.5);  mul_532 = None
    exp_18: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_533);  mul_533 = None
    mul_534: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_535: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_86, mul_534);  view_86 = mul_534 = None
    add_209: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_531, mul_535);  mul_531 = mul_535 = None
    mul_536: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_312, add_209);  view_312 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_313: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_536, [1568, 2048]);  mul_536 = None
    mm_76: "f32[1568, 512]" = torch.ops.aten.mm.default(view_313, permute_348);  permute_348 = None
    permute_349: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_313, [1, 0])
    mm_77: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_349, view_85);  permute_349 = view_85 = None
    permute_350: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_138: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
    view_314: "f32[2048]" = torch.ops.aten.reshape.default(sum_138, [2048]);  sum_138 = None
    permute_351: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_315: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_76, [8, 14, 14, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_538: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_315, primals_58);  primals_58 = None
    mul_539: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_538, 512)
    sum_139: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [3], True)
    mul_540: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_538, mul_108);  mul_538 = None
    sum_140: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [3], True);  mul_540 = None
    mul_541: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_108, sum_140);  sum_140 = None
    sub_102: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_539, sum_139);  mul_539 = sum_139 = None
    sub_103: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_102, mul_541);  sub_102 = mul_541 = None
    div_21: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 512);  rsqrt_20 = None
    mul_542: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_21, sub_103);  div_21 = sub_103 = None
    mul_543: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_315, mul_108);  mul_108 = None
    sum_141: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 1, 2]);  mul_543 = None
    sum_142: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 1, 2]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_352: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_542, [0, 3, 1, 2]);  mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(permute_352, add_73, primals_227, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_352 = add_73 = primals_227 = None
    getitem_139: "f32[8, 512, 14, 14]" = convolution_backward_19[0]
    getitem_140: "f32[512, 1, 7, 7]" = convolution_backward_19[1]
    getitem_141: "f32[512]" = convolution_backward_19[2];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_210: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_207, getitem_139);  add_207 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_544: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, permute_73);  permute_73 = None
    mul_545: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_210, view_84);  view_84 = None
    sum_143: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3], True);  mul_544 = None
    view_316: "f32[512]" = torch.ops.aten.reshape.default(sum_143, [512]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_353: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_545, [0, 2, 3, 1]);  mul_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_317: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_353, [1568, 512]);  permute_353 = None
    mm_78: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_317, permute_354);  permute_354 = None
    permute_355: "f32[512, 1568]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_79: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_355, view_82);  permute_355 = view_82 = None
    permute_356: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_144: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[512]" = torch.ops.aten.reshape.default(sum_144, [512]);  sum_144 = None
    permute_357: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_319: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_78, [8, 14, 14, 2048]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_547: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_548: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, view_81)
    mul_549: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_548, -0.5);  mul_548 = None
    exp_19: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_549);  mul_549 = None
    mul_550: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_551: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_81, mul_550);  view_81 = mul_550 = None
    add_212: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_547, mul_551);  mul_547 = mul_551 = None
    mul_552: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_319, add_212);  view_319 = add_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_320: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_552, [1568, 2048]);  mul_552 = None
    mm_80: "f32[1568, 512]" = torch.ops.aten.mm.default(view_320, permute_358);  permute_358 = None
    permute_359: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_81: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_359, view_80);  permute_359 = view_80 = None
    permute_360: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_145: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[2048]" = torch.ops.aten.reshape.default(sum_145, [2048]);  sum_145 = None
    permute_361: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    view_322: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_80, [8, 14, 14, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_554: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_322, primals_55);  primals_55 = None
    mul_555: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_554, 512)
    sum_146: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [3], True)
    mul_556: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_554, mul_102);  mul_554 = None
    sum_147: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [3], True);  mul_556 = None
    mul_557: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_102, sum_147);  sum_147 = None
    sub_105: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_555, sum_146);  mul_555 = sum_146 = None
    sub_106: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_105, mul_557);  sub_105 = mul_557 = None
    div_22: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 512);  rsqrt_19 = None
    mul_558: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_22, sub_106);  div_22 = sub_106 = None
    mul_559: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_322, mul_102);  mul_102 = None
    sum_148: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1, 2]);  mul_559 = None
    sum_149: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_322, [0, 1, 2]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_362: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_558, [0, 3, 1, 2]);  mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(permute_362, add_69, primals_221, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_362 = add_69 = primals_221 = None
    getitem_142: "f32[8, 512, 14, 14]" = convolution_backward_20[0]
    getitem_143: "f32[512, 1, 7, 7]" = convolution_backward_20[1]
    getitem_144: "f32[512]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_213: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_210, getitem_142);  add_210 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_560: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, permute_69);  permute_69 = None
    mul_561: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_213, view_79);  view_79 = None
    sum_150: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3], True);  mul_560 = None
    view_323: "f32[512]" = torch.ops.aten.reshape.default(sum_150, [512]);  sum_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_363: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_561, [0, 2, 3, 1]);  mul_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_324: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_363, [1568, 512]);  permute_363 = None
    mm_82: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_324, permute_364);  permute_364 = None
    permute_365: "f32[512, 1568]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_83: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_365, view_77);  permute_365 = view_77 = None
    permute_366: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_151: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[512]" = torch.ops.aten.reshape.default(sum_151, [512]);  sum_151 = None
    permute_367: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_326: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_82, [8, 14, 14, 2048]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_563: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
    mul_564: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, view_76)
    mul_565: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_564, -0.5);  mul_564 = None
    exp_20: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_565);  mul_565 = None
    mul_566: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_567: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_76, mul_566);  view_76 = mul_566 = None
    add_215: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_563, mul_567);  mul_563 = mul_567 = None
    mul_568: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_326, add_215);  view_326 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_327: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_568, [1568, 2048]);  mul_568 = None
    mm_84: "f32[1568, 512]" = torch.ops.aten.mm.default(view_327, permute_368);  permute_368 = None
    permute_369: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_85: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_369, view_75);  permute_369 = view_75 = None
    permute_370: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_152: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[2048]" = torch.ops.aten.reshape.default(sum_152, [2048]);  sum_152 = None
    permute_371: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_329: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_84, [8, 14, 14, 512]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_570: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_329, primals_52);  primals_52 = None
    mul_571: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_570, 512)
    sum_153: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [3], True)
    mul_572: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_570, mul_96);  mul_570 = None
    sum_154: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [3], True);  mul_572 = None
    mul_573: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_96, sum_154);  sum_154 = None
    sub_108: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_571, sum_153);  mul_571 = sum_153 = None
    sub_109: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_108, mul_573);  sub_108 = mul_573 = None
    div_23: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    mul_574: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_23, sub_109);  div_23 = sub_109 = None
    mul_575: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_329, mul_96);  mul_96 = None
    sum_155: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1, 2]);  mul_575 = None
    sum_156: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_329, [0, 1, 2]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_372: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_574, [0, 3, 1, 2]);  mul_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(permute_372, add_65, primals_215, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_372 = add_65 = primals_215 = None
    getitem_145: "f32[8, 512, 14, 14]" = convolution_backward_21[0]
    getitem_146: "f32[512, 1, 7, 7]" = convolution_backward_21[1]
    getitem_147: "f32[512]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_216: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_213, getitem_145);  add_213 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_576: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, permute_65);  permute_65 = None
    mul_577: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_216, view_74);  view_74 = None
    sum_157: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [0, 2, 3], True);  mul_576 = None
    view_330: "f32[512]" = torch.ops.aten.reshape.default(sum_157, [512]);  sum_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_373: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_577, [0, 2, 3, 1]);  mul_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_331: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_373, [1568, 512]);  permute_373 = None
    mm_86: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_331, permute_374);  permute_374 = None
    permute_375: "f32[512, 1568]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_87: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_375, view_72);  permute_375 = view_72 = None
    permute_376: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_158: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[512]" = torch.ops.aten.reshape.default(sum_158, [512]);  sum_158 = None
    permute_377: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_333: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_86, [8, 14, 14, 2048]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_579: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_580: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, view_71)
    mul_581: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_580, -0.5);  mul_580 = None
    exp_21: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_581);  mul_581 = None
    mul_582: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_583: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_71, mul_582);  view_71 = mul_582 = None
    add_218: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_579, mul_583);  mul_579 = mul_583 = None
    mul_584: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_333, add_218);  view_333 = add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_334: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_584, [1568, 2048]);  mul_584 = None
    mm_88: "f32[1568, 512]" = torch.ops.aten.mm.default(view_334, permute_378);  permute_378 = None
    permute_379: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_89: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_379, view_70);  permute_379 = view_70 = None
    permute_380: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_159: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[2048]" = torch.ops.aten.reshape.default(sum_159, [2048]);  sum_159 = None
    permute_381: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    view_336: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_88, [8, 14, 14, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_586: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_336, primals_49);  primals_49 = None
    mul_587: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_586, 512)
    sum_160: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [3], True)
    mul_588: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_586, mul_90);  mul_586 = None
    sum_161: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [3], True);  mul_588 = None
    mul_589: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_90, sum_161);  sum_161 = None
    sub_111: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_587, sum_160);  mul_587 = sum_160 = None
    sub_112: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_111, mul_589);  sub_111 = mul_589 = None
    div_24: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    mul_590: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_24, sub_112);  div_24 = sub_112 = None
    mul_591: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_336, mul_90);  mul_90 = None
    sum_162: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 1, 2]);  mul_591 = None
    sum_163: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 1, 2]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_382: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_590, [0, 3, 1, 2]);  mul_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(permute_382, add_61, primals_209, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_382 = add_61 = primals_209 = None
    getitem_148: "f32[8, 512, 14, 14]" = convolution_backward_22[0]
    getitem_149: "f32[512, 1, 7, 7]" = convolution_backward_22[1]
    getitem_150: "f32[512]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_219: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_216, getitem_148);  add_216 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_592: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, permute_61);  permute_61 = None
    mul_593: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, view_69);  view_69 = None
    sum_164: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3], True);  mul_592 = None
    view_337: "f32[512]" = torch.ops.aten.reshape.default(sum_164, [512]);  sum_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_383: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_593, [0, 2, 3, 1]);  mul_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_338: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_383, [1568, 512]);  permute_383 = None
    mm_90: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_338, permute_384);  permute_384 = None
    permute_385: "f32[512, 1568]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_91: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_385, view_67);  permute_385 = view_67 = None
    permute_386: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_165: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[512]" = torch.ops.aten.reshape.default(sum_165, [512]);  sum_165 = None
    permute_387: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    view_340: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_90, [8, 14, 14, 2048]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_595: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_60, 0.5);  add_60 = None
    mul_596: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, view_66)
    mul_597: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_596, -0.5);  mul_596 = None
    exp_22: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_597);  mul_597 = None
    mul_598: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_599: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_66, mul_598);  view_66 = mul_598 = None
    add_221: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_595, mul_599);  mul_595 = mul_599 = None
    mul_600: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_340, add_221);  view_340 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_341: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_600, [1568, 2048]);  mul_600 = None
    mm_92: "f32[1568, 512]" = torch.ops.aten.mm.default(view_341, permute_388);  permute_388 = None
    permute_389: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_93: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_389, view_65);  permute_389 = view_65 = None
    permute_390: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_166: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[2048]" = torch.ops.aten.reshape.default(sum_166, [2048]);  sum_166 = None
    permute_391: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_343: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_92, [8, 14, 14, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_602: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_343, primals_46);  primals_46 = None
    mul_603: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_602, 512)
    sum_167: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [3], True)
    mul_604: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_602, mul_84);  mul_602 = None
    sum_168: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_604, [3], True);  mul_604 = None
    mul_605: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_84, sum_168);  sum_168 = None
    sub_114: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_603, sum_167);  mul_603 = sum_167 = None
    sub_115: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_114, mul_605);  sub_114 = mul_605 = None
    div_25: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    mul_606: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_25, sub_115);  div_25 = sub_115 = None
    mul_607: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_343, mul_84);  mul_84 = None
    sum_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 1, 2]);  mul_607 = None
    sum_170: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1, 2]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_392: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_606, [0, 3, 1, 2]);  mul_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(permute_392, add_57, primals_203, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_392 = add_57 = primals_203 = None
    getitem_151: "f32[8, 512, 14, 14]" = convolution_backward_23[0]
    getitem_152: "f32[512, 1, 7, 7]" = convolution_backward_23[1]
    getitem_153: "f32[512]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_222: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_219, getitem_151);  add_219 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_608: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_222, permute_57);  permute_57 = None
    mul_609: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_222, view_64);  view_64 = None
    sum_171: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3], True);  mul_608 = None
    view_344: "f32[512]" = torch.ops.aten.reshape.default(sum_171, [512]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_393: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_609, [0, 2, 3, 1]);  mul_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_345: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_393, [1568, 512]);  permute_393 = None
    mm_94: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_345, permute_394);  permute_394 = None
    permute_395: "f32[512, 1568]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_95: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_395, view_62);  permute_395 = view_62 = None
    permute_396: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_172: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[512]" = torch.ops.aten.reshape.default(sum_172, [512]);  sum_172 = None
    permute_397: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_347: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_94, [8, 14, 14, 2048]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_611: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_612: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, view_61)
    mul_613: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_612, -0.5);  mul_612 = None
    exp_23: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_613);  mul_613 = None
    mul_614: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_615: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_61, mul_614);  view_61 = mul_614 = None
    add_224: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_611, mul_615);  mul_611 = mul_615 = None
    mul_616: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_347, add_224);  view_347 = add_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_348: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_616, [1568, 2048]);  mul_616 = None
    mm_96: "f32[1568, 512]" = torch.ops.aten.mm.default(view_348, permute_398);  permute_398 = None
    permute_399: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_97: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_399, view_60);  permute_399 = view_60 = None
    permute_400: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_173: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
    view_349: "f32[2048]" = torch.ops.aten.reshape.default(sum_173, [2048]);  sum_173 = None
    permute_401: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_350: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_96, [8, 14, 14, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_618: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_350, primals_43);  primals_43 = None
    mul_619: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_618, 512)
    sum_174: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [3], True)
    mul_620: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_618, mul_78);  mul_618 = None
    sum_175: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [3], True);  mul_620 = None
    mul_621: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_78, sum_175);  sum_175 = None
    sub_117: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_619, sum_174);  mul_619 = sum_174 = None
    sub_118: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_117, mul_621);  sub_117 = mul_621 = None
    div_26: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    mul_622: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_26, sub_118);  div_26 = sub_118 = None
    mul_623: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_350, mul_78);  mul_78 = None
    sum_176: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1, 2]);  mul_623 = None
    sum_177: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_350, [0, 1, 2]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_402: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_622, [0, 3, 1, 2]);  mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(permute_402, add_53, primals_197, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_402 = add_53 = primals_197 = None
    getitem_154: "f32[8, 512, 14, 14]" = convolution_backward_24[0]
    getitem_155: "f32[512, 1, 7, 7]" = convolution_backward_24[1]
    getitem_156: "f32[512]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_225: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_222, getitem_154);  add_222 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_624: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, permute_53);  permute_53 = None
    mul_625: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_225, view_59);  view_59 = None
    sum_178: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3], True);  mul_624 = None
    view_351: "f32[512]" = torch.ops.aten.reshape.default(sum_178, [512]);  sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_403: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_625, [0, 2, 3, 1]);  mul_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_352: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_403, [1568, 512]);  permute_403 = None
    mm_98: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_352, permute_404);  permute_404 = None
    permute_405: "f32[512, 1568]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_405, view_57);  permute_405 = view_57 = None
    permute_406: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_179: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[512]" = torch.ops.aten.reshape.default(sum_179, [512]);  sum_179 = None
    permute_407: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_354: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_98, [8, 14, 14, 2048]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_627: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_52, 0.5);  add_52 = None
    mul_628: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, view_56)
    mul_629: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_628, -0.5);  mul_628 = None
    exp_24: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_629);  mul_629 = None
    mul_630: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_631: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_56, mul_630);  view_56 = mul_630 = None
    add_227: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_627, mul_631);  mul_627 = mul_631 = None
    mul_632: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_354, add_227);  view_354 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_355: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_632, [1568, 2048]);  mul_632 = None
    mm_100: "f32[1568, 512]" = torch.ops.aten.mm.default(view_355, permute_408);  permute_408 = None
    permute_409: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_409, view_55);  permute_409 = view_55 = None
    permute_410: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_180: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[2048]" = torch.ops.aten.reshape.default(sum_180, [2048]);  sum_180 = None
    permute_411: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_357: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_100, [8, 14, 14, 512]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_634: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_357, primals_40);  primals_40 = None
    mul_635: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_634, 512)
    sum_181: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [3], True)
    mul_636: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_634, mul_72);  mul_634 = None
    sum_182: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [3], True);  mul_636 = None
    mul_637: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_72, sum_182);  sum_182 = None
    sub_120: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_635, sum_181);  mul_635 = sum_181 = None
    sub_121: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_120, mul_637);  sub_120 = mul_637 = None
    div_27: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    mul_638: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_27, sub_121);  div_27 = sub_121 = None
    mul_639: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_357, mul_72);  mul_72 = None
    sum_183: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 1, 2]);  mul_639 = None
    sum_184: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_357, [0, 1, 2]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_412: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_638, [0, 3, 1, 2]);  mul_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(permute_412, add_49, primals_191, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_412 = add_49 = primals_191 = None
    getitem_157: "f32[8, 512, 14, 14]" = convolution_backward_25[0]
    getitem_158: "f32[512, 1, 7, 7]" = convolution_backward_25[1]
    getitem_159: "f32[512]" = convolution_backward_25[2];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_228: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_225, getitem_157);  add_225 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_640: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_228, permute_49);  permute_49 = None
    mul_641: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_228, view_54);  view_54 = None
    sum_185: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3], True);  mul_640 = None
    view_358: "f32[512]" = torch.ops.aten.reshape.default(sum_185, [512]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_413: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_641, [0, 2, 3, 1]);  mul_641 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_413, [1568, 512]);  permute_413 = None
    mm_102: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_359, permute_414);  permute_414 = None
    permute_415: "f32[512, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_103: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_415, view_52);  permute_415 = view_52 = None
    permute_416: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_186: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[512]" = torch.ops.aten.reshape.default(sum_186, [512]);  sum_186 = None
    permute_417: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_361: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_102, [8, 14, 14, 2048]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_643: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_644: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_645: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_644, -0.5);  mul_644 = None
    exp_25: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_645);  mul_645 = None
    mul_646: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_647: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_51, mul_646);  view_51 = mul_646 = None
    add_230: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_643, mul_647);  mul_643 = mul_647 = None
    mul_648: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_361, add_230);  view_361 = add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_362: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_648, [1568, 2048]);  mul_648 = None
    mm_104: "f32[1568, 512]" = torch.ops.aten.mm.default(view_362, permute_418);  permute_418 = None
    permute_419: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_105: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_419, view_50);  permute_419 = view_50 = None
    permute_420: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_187: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[2048]" = torch.ops.aten.reshape.default(sum_187, [2048]);  sum_187 = None
    permute_421: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
    view_364: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_104, [8, 14, 14, 512]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_650: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_364, primals_37);  primals_37 = None
    mul_651: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_650, 512)
    sum_188: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_650, [3], True)
    mul_652: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_650, mul_66);  mul_650 = None
    sum_189: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [3], True);  mul_652 = None
    mul_653: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_66, sum_189);  sum_189 = None
    sub_123: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_651, sum_188);  mul_651 = sum_188 = None
    sub_124: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_123, mul_653);  sub_123 = mul_653 = None
    div_28: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    mul_654: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_28, sub_124);  div_28 = sub_124 = None
    mul_655: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_364, mul_66);  mul_66 = None
    sum_190: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 1, 2]);  mul_655 = None
    sum_191: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_364, [0, 1, 2]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_422: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_654, [0, 3, 1, 2]);  mul_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(permute_422, add_45, primals_185, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_422 = add_45 = primals_185 = None
    getitem_160: "f32[8, 512, 14, 14]" = convolution_backward_26[0]
    getitem_161: "f32[512, 1, 7, 7]" = convolution_backward_26[1]
    getitem_162: "f32[512]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_231: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_228, getitem_160);  add_228 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_656: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, permute_45);  permute_45 = None
    mul_657: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_231, view_49);  view_49 = None
    sum_192: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_656, [0, 2, 3], True);  mul_656 = None
    view_365: "f32[512]" = torch.ops.aten.reshape.default(sum_192, [512]);  sum_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_423: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_657, [0, 2, 3, 1]);  mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_366: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_423, [1568, 512]);  permute_423 = None
    mm_106: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_366, permute_424);  permute_424 = None
    permute_425: "f32[512, 1568]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_107: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_47);  permute_425 = view_47 = None
    permute_426: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_193: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[512]" = torch.ops.aten.reshape.default(sum_193, [512]);  sum_193 = None
    permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_426, [1, 0]);  permute_426 = None
    view_368: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_106, [8, 14, 14, 2048]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_659: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
    mul_660: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_661: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_660, -0.5);  mul_660 = None
    exp_26: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_661);  mul_661 = None
    mul_662: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_663: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_46, mul_662);  view_46 = mul_662 = None
    add_233: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_659, mul_663);  mul_659 = mul_663 = None
    mul_664: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_368, add_233);  view_368 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_369: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_664, [1568, 2048]);  mul_664 = None
    mm_108: "f32[1568, 512]" = torch.ops.aten.mm.default(view_369, permute_428);  permute_428 = None
    permute_429: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_369, [1, 0])
    mm_109: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_45);  permute_429 = view_45 = None
    permute_430: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_194: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
    view_370: "f32[2048]" = torch.ops.aten.reshape.default(sum_194, [2048]);  sum_194 = None
    permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_430, [1, 0]);  permute_430 = None
    view_371: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_108, [8, 14, 14, 512]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_666: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_371, primals_34);  primals_34 = None
    mul_667: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_666, 512)
    sum_195: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [3], True)
    mul_668: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_666, mul_60);  mul_666 = None
    sum_196: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [3], True);  mul_668 = None
    mul_669: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_60, sum_196);  sum_196 = None
    sub_126: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_667, sum_195);  mul_667 = sum_195 = None
    sub_127: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_126, mul_669);  sub_126 = mul_669 = None
    div_29: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    mul_670: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_29, sub_127);  div_29 = sub_127 = None
    mul_671: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_371, mul_60);  mul_60 = None
    sum_197: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1, 2]);  mul_671 = None
    sum_198: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_371, [0, 1, 2]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_432: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_670, [0, 3, 1, 2]);  mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(permute_432, add_41, primals_179, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_432 = add_41 = primals_179 = None
    getitem_163: "f32[8, 512, 14, 14]" = convolution_backward_27[0]
    getitem_164: "f32[512, 1, 7, 7]" = convolution_backward_27[1]
    getitem_165: "f32[512]" = convolution_backward_27[2];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_234: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_231, getitem_163);  add_231 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_672: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, permute_41);  permute_41 = None
    mul_673: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_234, view_44);  view_44 = None
    sum_199: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3], True);  mul_672 = None
    view_372: "f32[512]" = torch.ops.aten.reshape.default(sum_199, [512]);  sum_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_433: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_673, [0, 2, 3, 1]);  mul_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_373: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_433, [1568, 512]);  permute_433 = None
    mm_110: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_373, permute_434);  permute_434 = None
    permute_435: "f32[512, 1568]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_111: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_435, view_42);  permute_435 = view_42 = None
    permute_436: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_200: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
    view_374: "f32[512]" = torch.ops.aten.reshape.default(sum_200, [512]);  sum_200 = None
    permute_437: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_375: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_110, [8, 14, 14, 2048]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_675: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_676: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_677: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_676, -0.5);  mul_676 = None
    exp_27: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_677);  mul_677 = None
    mul_678: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_679: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_41, mul_678);  view_41 = mul_678 = None
    add_236: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_675, mul_679);  mul_675 = mul_679 = None
    mul_680: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_375, add_236);  view_375 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_376: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_680, [1568, 2048]);  mul_680 = None
    mm_112: "f32[1568, 512]" = torch.ops.aten.mm.default(view_376, permute_438);  permute_438 = None
    permute_439: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_113: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_439, view_40);  permute_439 = view_40 = None
    permute_440: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_201: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_376, [0], True);  view_376 = None
    view_377: "f32[2048]" = torch.ops.aten.reshape.default(sum_201, [2048]);  sum_201 = None
    permute_441: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_378: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_112, [8, 14, 14, 512]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_682: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_378, primals_31);  primals_31 = None
    mul_683: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_682, 512)
    sum_202: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [3], True)
    mul_684: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_682, mul_54);  mul_682 = None
    sum_203: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [3], True);  mul_684 = None
    mul_685: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_54, sum_203);  sum_203 = None
    sub_129: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_683, sum_202);  mul_683 = sum_202 = None
    sub_130: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_129, mul_685);  sub_129 = mul_685 = None
    div_30: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    mul_686: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_30, sub_130);  div_30 = sub_130 = None
    mul_687: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_378, mul_54);  mul_54 = None
    sum_204: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 1, 2]);  mul_687 = None
    sum_205: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_378, [0, 1, 2]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_442: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_686, [0, 3, 1, 2]);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(permute_442, add_37, primals_173, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_442 = add_37 = primals_173 = None
    getitem_166: "f32[8, 512, 14, 14]" = convolution_backward_28[0]
    getitem_167: "f32[512, 1, 7, 7]" = convolution_backward_28[1]
    getitem_168: "f32[512]" = convolution_backward_28[2];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_237: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_234, getitem_166);  add_234 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_688: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_237, permute_37);  permute_37 = None
    mul_689: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_237, view_39);  view_39 = None
    sum_206: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3], True);  mul_688 = None
    view_379: "f32[512]" = torch.ops.aten.reshape.default(sum_206, [512]);  sum_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_443: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_689, [0, 2, 3, 1]);  mul_689 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_380: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_443, [1568, 512]);  permute_443 = None
    mm_114: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_380, permute_444);  permute_444 = None
    permute_445: "f32[512, 1568]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_115: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_445, view_37);  permute_445 = view_37 = None
    permute_446: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_207: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[512]" = torch.ops.aten.reshape.default(sum_207, [512]);  sum_207 = None
    permute_447: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_382: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_114, [8, 14, 14, 2048]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_691: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_36, 0.5);  add_36 = None
    mul_692: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, view_36)
    mul_693: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_692, -0.5);  mul_692 = None
    exp_28: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_693);  mul_693 = None
    mul_694: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_695: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_36, mul_694);  view_36 = mul_694 = None
    add_239: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_691, mul_695);  mul_691 = mul_695 = None
    mul_696: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_382, add_239);  view_382 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_383: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_696, [1568, 2048]);  mul_696 = None
    mm_116: "f32[1568, 512]" = torch.ops.aten.mm.default(view_383, permute_448);  permute_448 = None
    permute_449: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_117: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_449, view_35);  permute_449 = view_35 = None
    permute_450: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_208: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[2048]" = torch.ops.aten.reshape.default(sum_208, [2048]);  sum_208 = None
    permute_451: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_385: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_116, [8, 14, 14, 512]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_698: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_385, primals_28);  primals_28 = None
    mul_699: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, 512)
    sum_209: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [3], True)
    mul_700: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_698, mul_48);  mul_698 = None
    sum_210: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [3], True);  mul_700 = None
    mul_701: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_48, sum_210);  sum_210 = None
    sub_132: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_699, sum_209);  mul_699 = sum_209 = None
    sub_133: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_132, mul_701);  sub_132 = mul_701 = None
    div_31: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    mul_702: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_31, sub_133);  div_31 = sub_133 = None
    mul_703: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_385, mul_48);  mul_48 = None
    sum_211: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1, 2]);  mul_703 = None
    sum_212: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_385, [0, 1, 2]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_452: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_702, [0, 3, 1, 2]);  mul_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(permute_452, add_33, primals_167, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_452 = add_33 = primals_167 = None
    getitem_169: "f32[8, 512, 14, 14]" = convolution_backward_29[0]
    getitem_170: "f32[512, 1, 7, 7]" = convolution_backward_29[1]
    getitem_171: "f32[512]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_240: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_237, getitem_169);  add_237 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_704: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, permute_33);  permute_33 = None
    mul_705: "f32[8, 512, 14, 14]" = torch.ops.aten.mul.Tensor(add_240, view_34);  view_34 = None
    sum_213: "f32[1, 512, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3], True);  mul_704 = None
    view_386: "f32[512]" = torch.ops.aten.reshape.default(sum_213, [512]);  sum_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_453: "f32[8, 14, 14, 512]" = torch.ops.aten.permute.default(mul_705, [0, 2, 3, 1]);  mul_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1568, 512]" = torch.ops.aten.reshape.default(permute_453, [1568, 512]);  permute_453 = None
    mm_118: "f32[1568, 2048]" = torch.ops.aten.mm.default(view_387, permute_454);  permute_454 = None
    permute_455: "f32[512, 1568]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_455, view_32);  permute_455 = view_32 = None
    permute_456: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_214: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[512]" = torch.ops.aten.reshape.default(sum_214, [512]);  sum_214 = None
    permute_457: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_389: "f32[8, 14, 14, 2048]" = torch.ops.aten.reshape.default(mm_118, [8, 14, 14, 2048]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_707: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_708: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, view_31)
    mul_709: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(mul_708, -0.5);  mul_708 = None
    exp_29: "f32[8, 14, 14, 2048]" = torch.ops.aten.exp.default(mul_709);  mul_709 = None
    mul_710: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_711: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_31, mul_710);  view_31 = mul_710 = None
    add_242: "f32[8, 14, 14, 2048]" = torch.ops.aten.add.Tensor(mul_707, mul_711);  mul_707 = mul_711 = None
    mul_712: "f32[8, 14, 14, 2048]" = torch.ops.aten.mul.Tensor(view_389, add_242);  view_389 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1568, 2048]" = torch.ops.aten.reshape.default(mul_712, [1568, 2048]);  mul_712 = None
    mm_120: "f32[1568, 512]" = torch.ops.aten.mm.default(view_390, permute_458);  permute_458 = None
    permute_459: "f32[2048, 1568]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_459, view_30);  permute_459 = view_30 = None
    permute_460: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_215: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[2048]" = torch.ops.aten.reshape.default(sum_215, [2048]);  sum_215 = None
    permute_461: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    view_392: "f32[8, 14, 14, 512]" = torch.ops.aten.reshape.default(mm_120, [8, 14, 14, 512]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_714: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_392, primals_25);  primals_25 = None
    mul_715: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_714, 512)
    sum_216: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [3], True)
    mul_716: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_714, mul_42);  mul_714 = None
    sum_217: "f32[8, 14, 14, 1]" = torch.ops.aten.sum.dim_IntList(mul_716, [3], True);  mul_716 = None
    mul_717: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(mul_42, sum_217);  sum_217 = None
    sub_135: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(mul_715, sum_216);  mul_715 = sum_216 = None
    sub_136: "f32[8, 14, 14, 512]" = torch.ops.aten.sub.Tensor(sub_135, mul_717);  sub_135 = mul_717 = None
    div_32: "f32[8, 14, 14, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    mul_718: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(div_32, sub_136);  div_32 = sub_136 = None
    mul_719: "f32[8, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_392, mul_42);  mul_42 = None
    sum_218: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1, 2]);  mul_719 = None
    sum_219: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_392, [0, 1, 2]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_462: "f32[8, 512, 14, 14]" = torch.ops.aten.permute.default(mul_718, [0, 3, 1, 2]);  mul_718 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(permute_462, convolution_8, primals_161, [512], [1, 1], [3, 3], [1, 1], False, [0, 0], 512, [True, True, True]);  permute_462 = convolution_8 = primals_161 = None
    getitem_172: "f32[8, 512, 14, 14]" = convolution_backward_30[0]
    getitem_173: "f32[512, 1, 7, 7]" = convolution_backward_30[1]
    getitem_174: "f32[512]" = convolution_backward_30[2];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_243: "f32[8, 512, 14, 14]" = torch.ops.aten.add.Tensor(add_240, getitem_172);  add_240 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_243, permute_29, primals_159, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  add_243 = permute_29 = primals_159 = None
    getitem_175: "f32[8, 256, 28, 28]" = convolution_backward_31[0]
    getitem_176: "f32[512, 256, 2, 2]" = convolution_backward_31[1]
    getitem_177: "f32[512]" = convolution_backward_31[2];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_463: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 3, 1]);  getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_721: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(permute_463, primals_23);  primals_23 = None
    mul_722: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_721, 256)
    sum_220: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [3], True)
    mul_723: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_721, mul_40);  mul_721 = None
    sum_221: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_723, [3], True);  mul_723 = None
    mul_724: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_40, sum_221);  sum_221 = None
    sub_138: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_722, sum_220);  mul_722 = sum_220 = None
    sub_139: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_138, mul_724);  sub_138 = mul_724 = None
    mul_725: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_139);  div_33 = sub_139 = None
    mul_726: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(permute_463, mul_40);  mul_40 = None
    sum_222: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 1, 2]);  mul_726 = None
    sum_223: "f32[256]" = torch.ops.aten.sum.dim_IntList(permute_463, [0, 1, 2]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_464: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_725, [0, 3, 1, 2]);  mul_725 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_727: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_464, permute_27);  permute_27 = None
    mul_728: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(permute_464, view_29);  view_29 = None
    sum_224: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 2, 3], True);  mul_727 = None
    view_393: "f32[256]" = torch.ops.aten.reshape.default(sum_224, [256]);  sum_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_465: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_728, [0, 2, 3, 1]);  mul_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_394: "f32[6272, 256]" = torch.ops.aten.reshape.default(permute_465, [6272, 256]);  permute_465 = None
    mm_122: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_394, permute_466);  permute_466 = None
    permute_467: "f32[256, 6272]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_123: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_467, view_27);  permute_467 = view_27 = None
    permute_468: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_225: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[256]" = torch.ops.aten.reshape.default(sum_225, [256]);  sum_225 = None
    permute_469: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
    view_396: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(mm_122, [8, 28, 28, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_730: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
    mul_731: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, view_26)
    mul_732: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_731, -0.5);  mul_731 = None
    exp_30: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_732);  mul_732 = None
    mul_733: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_734: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_26, mul_733);  view_26 = mul_733 = None
    add_245: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_730, mul_734);  mul_730 = mul_734 = None
    mul_735: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_396, add_245);  view_396 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_397: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_735, [6272, 1024]);  mul_735 = None
    mm_124: "f32[6272, 256]" = torch.ops.aten.mm.default(view_397, permute_470);  permute_470 = None
    permute_471: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_125: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_471, view_25);  permute_471 = view_25 = None
    permute_472: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_226: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[1024]" = torch.ops.aten.reshape.default(sum_226, [1024]);  sum_226 = None
    permute_473: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_399: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(mm_124, [8, 28, 28, 256]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_737: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_399, primals_20);  primals_20 = None
    mul_738: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_737, 256)
    sum_227: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [3], True)
    mul_739: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_737, mul_34);  mul_737 = None
    sum_228: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_739, [3], True);  mul_739 = None
    mul_740: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_34, sum_228);  sum_228 = None
    sub_141: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_738, sum_227);  mul_738 = sum_227 = None
    sub_142: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_141, mul_740);  sub_141 = mul_740 = None
    div_34: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_741: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_142);  div_34 = sub_142 = None
    mul_742: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_399, mul_34);  mul_34 = None
    sum_229: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 1, 2]);  mul_742 = None
    sum_230: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_399, [0, 1, 2]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_474: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_741, [0, 3, 1, 2]);  mul_741 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(permute_474, add_23, primals_153, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True]);  permute_474 = add_23 = primals_153 = None
    getitem_178: "f32[8, 256, 28, 28]" = convolution_backward_32[0]
    getitem_179: "f32[256, 1, 7, 7]" = convolution_backward_32[1]
    getitem_180: "f32[256]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_246: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(permute_464, getitem_178);  permute_464 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_743: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_246, permute_23);  permute_23 = None
    mul_744: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_246, view_24);  view_24 = None
    sum_231: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3], True);  mul_743 = None
    view_400: "f32[256]" = torch.ops.aten.reshape.default(sum_231, [256]);  sum_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_475: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_744, [0, 2, 3, 1]);  mul_744 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_401: "f32[6272, 256]" = torch.ops.aten.reshape.default(permute_475, [6272, 256]);  permute_475 = None
    mm_126: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_401, permute_476);  permute_476 = None
    permute_477: "f32[256, 6272]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_127: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_477, view_22);  permute_477 = view_22 = None
    permute_478: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_232: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[256]" = torch.ops.aten.reshape.default(sum_232, [256]);  sum_232 = None
    permute_479: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_403: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(mm_126, [8, 28, 28, 1024]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_746: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
    mul_747: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_748: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_747, -0.5);  mul_747 = None
    exp_31: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_748);  mul_748 = None
    mul_749: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_750: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_749);  view_21 = mul_749 = None
    add_248: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_746, mul_750);  mul_746 = mul_750 = None
    mul_751: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_403, add_248);  view_403 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_404: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_751, [6272, 1024]);  mul_751 = None
    mm_128: "f32[6272, 256]" = torch.ops.aten.mm.default(view_404, permute_480);  permute_480 = None
    permute_481: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_129: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_481, view_20);  permute_481 = view_20 = None
    permute_482: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_233: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[1024]" = torch.ops.aten.reshape.default(sum_233, [1024]);  sum_233 = None
    permute_483: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_406: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(mm_128, [8, 28, 28, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_753: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_406, primals_17);  primals_17 = None
    mul_754: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_753, 256)
    sum_234: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_753, [3], True)
    mul_755: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_753, mul_28);  mul_753 = None
    sum_235: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_755, [3], True);  mul_755 = None
    mul_756: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_28, sum_235);  sum_235 = None
    sub_144: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_754, sum_234);  mul_754 = sum_234 = None
    sub_145: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_144, mul_756);  sub_144 = mul_756 = None
    div_35: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_757: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_35, sub_145);  div_35 = sub_145 = None
    mul_758: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_406, mul_28);  mul_28 = None
    sum_236: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_758, [0, 1, 2]);  mul_758 = None
    sum_237: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1, 2]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_484: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_757, [0, 3, 1, 2]);  mul_757 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(permute_484, add_19, primals_147, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True]);  permute_484 = add_19 = primals_147 = None
    getitem_181: "f32[8, 256, 28, 28]" = convolution_backward_33[0]
    getitem_182: "f32[256, 1, 7, 7]" = convolution_backward_33[1]
    getitem_183: "f32[256]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_249: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_246, getitem_181);  add_246 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_759: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_249, permute_19);  permute_19 = None
    mul_760: "f32[8, 256, 28, 28]" = torch.ops.aten.mul.Tensor(add_249, view_19);  view_19 = None
    sum_238: "f32[1, 256, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3], True);  mul_759 = None
    view_407: "f32[256]" = torch.ops.aten.reshape.default(sum_238, [256]);  sum_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_485: "f32[8, 28, 28, 256]" = torch.ops.aten.permute.default(mul_760, [0, 2, 3, 1]);  mul_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_408: "f32[6272, 256]" = torch.ops.aten.reshape.default(permute_485, [6272, 256]);  permute_485 = None
    mm_130: "f32[6272, 1024]" = torch.ops.aten.mm.default(view_408, permute_486);  permute_486 = None
    permute_487: "f32[256, 6272]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_131: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_487, view_17);  permute_487 = view_17 = None
    permute_488: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_239: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[256]" = torch.ops.aten.reshape.default(sum_239, [256]);  sum_239 = None
    permute_489: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    view_410: "f32[8, 28, 28, 1024]" = torch.ops.aten.reshape.default(mm_130, [8, 28, 28, 1024]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_762: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
    mul_763: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, view_16)
    mul_764: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(mul_763, -0.5);  mul_763 = None
    exp_32: "f32[8, 28, 28, 1024]" = torch.ops.aten.exp.default(mul_764);  mul_764 = None
    mul_765: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_766: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_16, mul_765);  view_16 = mul_765 = None
    add_251: "f32[8, 28, 28, 1024]" = torch.ops.aten.add.Tensor(mul_762, mul_766);  mul_762 = mul_766 = None
    mul_767: "f32[8, 28, 28, 1024]" = torch.ops.aten.mul.Tensor(view_410, add_251);  view_410 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_411: "f32[6272, 1024]" = torch.ops.aten.reshape.default(mul_767, [6272, 1024]);  mul_767 = None
    mm_132: "f32[6272, 256]" = torch.ops.aten.mm.default(view_411, permute_490);  permute_490 = None
    permute_491: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_133: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_491, view_15);  permute_491 = view_15 = None
    permute_492: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[1024]" = torch.ops.aten.reshape.default(sum_240, [1024]);  sum_240 = None
    permute_493: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_492, [1, 0]);  permute_492 = None
    view_413: "f32[8, 28, 28, 256]" = torch.ops.aten.reshape.default(mm_132, [8, 28, 28, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_769: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_413, primals_14);  primals_14 = None
    mul_770: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_769, 256)
    sum_241: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_769, [3], True)
    mul_771: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_769, mul_22);  mul_769 = None
    sum_242: "f32[8, 28, 28, 1]" = torch.ops.aten.sum.dim_IntList(mul_771, [3], True);  mul_771 = None
    mul_772: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_242);  sum_242 = None
    sub_147: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(mul_770, sum_241);  mul_770 = sum_241 = None
    sub_148: "f32[8, 28, 28, 256]" = torch.ops.aten.sub.Tensor(sub_147, mul_772);  sub_147 = mul_772 = None
    div_36: "f32[8, 28, 28, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_773: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_148);  div_36 = sub_148 = None
    mul_774: "f32[8, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_413, mul_22);  mul_22 = None
    sum_243: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_774, [0, 1, 2]);  mul_774 = None
    sum_244: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_413, [0, 1, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_494: "f32[8, 256, 28, 28]" = torch.ops.aten.permute.default(mul_773, [0, 3, 1, 2]);  mul_773 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(permute_494, convolution_4, primals_141, [256], [1, 1], [3, 3], [1, 1], False, [0, 0], 256, [True, True, True]);  permute_494 = convolution_4 = primals_141 = None
    getitem_184: "f32[8, 256, 28, 28]" = convolution_backward_34[0]
    getitem_185: "f32[256, 1, 7, 7]" = convolution_backward_34[1]
    getitem_186: "f32[256]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_252: "f32[8, 256, 28, 28]" = torch.ops.aten.add.Tensor(add_249, getitem_184);  add_249 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:229, code: x = self.downsample(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_252, permute_15, primals_139, [256], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  add_252 = permute_15 = primals_139 = None
    getitem_187: "f32[8, 128, 56, 56]" = convolution_backward_35[0]
    getitem_188: "f32[256, 128, 2, 2]" = convolution_backward_35[1]
    getitem_189: "f32[256]" = convolution_backward_35[2];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_495: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 3, 1]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_776: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_495, primals_12);  primals_12 = None
    mul_777: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_776, 128)
    sum_245: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_776, [3], True)
    mul_778: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_776, mul_20);  mul_776 = None
    sum_246: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [3], True);  mul_778 = None
    mul_779: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_20, sum_246);  sum_246 = None
    sub_150: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_777, sum_245);  mul_777 = sum_245 = None
    sub_151: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_150, mul_779);  sub_150 = mul_779 = None
    mul_780: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_37, sub_151);  div_37 = sub_151 = None
    mul_781: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_495, mul_20);  mul_20 = None
    sum_247: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 1, 2]);  mul_781 = None
    sum_248: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_495, [0, 1, 2]);  permute_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_496: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_780, [0, 3, 1, 2]);  mul_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_782: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_496, permute_13);  permute_13 = None
    mul_783: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(permute_496, view_14);  view_14 = None
    sum_249: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3], True);  mul_782 = None
    view_414: "f32[128]" = torch.ops.aten.reshape.default(sum_249, [128]);  sum_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_497: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_783, [0, 2, 3, 1]);  mul_783 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_415: "f32[25088, 128]" = torch.ops.aten.reshape.default(permute_497, [25088, 128]);  permute_497 = None
    mm_134: "f32[25088, 512]" = torch.ops.aten.mm.default(view_415, permute_498);  permute_498 = None
    permute_499: "f32[128, 25088]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_135: "f32[128, 512]" = torch.ops.aten.mm.default(permute_499, view_12);  permute_499 = view_12 = None
    permute_500: "f32[512, 128]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_250: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[128]" = torch.ops.aten.reshape.default(sum_250, [128]);  sum_250 = None
    permute_501: "f32[128, 512]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_417: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(mm_134, [8, 56, 56, 512]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_785: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
    mul_786: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, view_11)
    mul_787: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_786, -0.5);  mul_786 = None
    exp_33: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_787);  mul_787 = None
    mul_788: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_789: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_11, mul_788);  view_11 = mul_788 = None
    add_254: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_785, mul_789);  mul_785 = mul_789 = None
    mul_790: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_417, add_254);  view_417 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_418: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_790, [25088, 512]);  mul_790 = None
    mm_136: "f32[25088, 128]" = torch.ops.aten.mm.default(view_418, permute_502);  permute_502 = None
    permute_503: "f32[512, 25088]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_137: "f32[512, 128]" = torch.ops.aten.mm.default(permute_503, view_10);  permute_503 = view_10 = None
    permute_504: "f32[128, 512]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_251: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
    view_419: "f32[512]" = torch.ops.aten.reshape.default(sum_251, [512]);  sum_251 = None
    permute_505: "f32[512, 128]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_420: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(mm_136, [8, 56, 56, 128]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_792: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_420, primals_9);  primals_9 = None
    mul_793: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_792, 128)
    sum_252: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_792, [3], True)
    mul_794: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_792, mul_14);  mul_792 = None
    sum_253: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_794, [3], True);  mul_794 = None
    mul_795: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_14, sum_253);  sum_253 = None
    sub_153: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_793, sum_252);  mul_793 = sum_252 = None
    sub_154: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_153, mul_795);  sub_153 = mul_795 = None
    div_38: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 128);  rsqrt_3 = None
    mul_796: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_38, sub_154);  div_38 = sub_154 = None
    mul_797: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_420, mul_14);  mul_14 = None
    sum_254: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 1, 2]);  mul_797 = None
    sum_255: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_420, [0, 1, 2]);  view_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_506: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_796, [0, 3, 1, 2]);  mul_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(permute_506, add_9, primals_133, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True]);  permute_506 = add_9 = primals_133 = None
    getitem_190: "f32[8, 128, 56, 56]" = convolution_backward_36[0]
    getitem_191: "f32[128, 1, 7, 7]" = convolution_backward_36[1]
    getitem_192: "f32[128]" = convolution_backward_36[2];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_255: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(permute_496, getitem_190);  permute_496 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_798: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_255, permute_9);  permute_9 = None
    mul_799: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_255, view_9);  view_9 = None
    sum_256: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3], True);  mul_798 = None
    view_421: "f32[128]" = torch.ops.aten.reshape.default(sum_256, [128]);  sum_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_507: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_799, [0, 2, 3, 1]);  mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_422: "f32[25088, 128]" = torch.ops.aten.reshape.default(permute_507, [25088, 128]);  permute_507 = None
    mm_138: "f32[25088, 512]" = torch.ops.aten.mm.default(view_422, permute_508);  permute_508 = None
    permute_509: "f32[128, 25088]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_139: "f32[128, 512]" = torch.ops.aten.mm.default(permute_509, view_7);  permute_509 = view_7 = None
    permute_510: "f32[512, 128]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_257: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[128]" = torch.ops.aten.reshape.default(sum_257, [128]);  sum_257 = None
    permute_511: "f32[128, 512]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_424: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(mm_138, [8, 56, 56, 512]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_801: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_802: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, view_6)
    mul_803: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_802, -0.5);  mul_802 = None
    exp_34: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_803);  mul_803 = None
    mul_804: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
    mul_805: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_6, mul_804);  view_6 = mul_804 = None
    add_257: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_801, mul_805);  mul_801 = mul_805 = None
    mul_806: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_424, add_257);  view_424 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_425: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_806, [25088, 512]);  mul_806 = None
    mm_140: "f32[25088, 128]" = torch.ops.aten.mm.default(view_425, permute_512);  permute_512 = None
    permute_513: "f32[512, 25088]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_141: "f32[512, 128]" = torch.ops.aten.mm.default(permute_513, view_5);  permute_513 = view_5 = None
    permute_514: "f32[128, 512]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_258: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[512]" = torch.ops.aten.reshape.default(sum_258, [512]);  sum_258 = None
    permute_515: "f32[512, 128]" = torch.ops.aten.permute.default(permute_514, [1, 0]);  permute_514 = None
    view_427: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(mm_140, [8, 56, 56, 128]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_808: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_427, primals_6);  primals_6 = None
    mul_809: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_808, 128)
    sum_259: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [3], True)
    mul_810: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_808, mul_8);  mul_808 = None
    sum_260: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_810, [3], True);  mul_810 = None
    mul_811: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_8, sum_260);  sum_260 = None
    sub_156: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_809, sum_259);  mul_809 = sum_259 = None
    sub_157: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_156, mul_811);  sub_156 = mul_811 = None
    div_39: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 128);  rsqrt_2 = None
    mul_812: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_39, sub_157);  div_39 = sub_157 = None
    mul_813: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_427, mul_8);  mul_8 = None
    sum_261: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_813, [0, 1, 2]);  mul_813 = None
    sum_262: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_427, [0, 1, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_516: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_812, [0, 3, 1, 2]);  mul_812 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(permute_516, add_5, primals_127, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True]);  permute_516 = add_5 = primals_127 = None
    getitem_193: "f32[8, 128, 56, 56]" = convolution_backward_37[0]
    getitem_194: "f32[128, 1, 7, 7]" = convolution_backward_37[1]
    getitem_195: "f32[128]" = convolution_backward_37[2];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_258: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_255, getitem_193);  add_255 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:162, code: x = x.mul(self.gamma.reshape(1, -1, 1, 1))
    mul_814: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_258, permute_5);  permute_5 = None
    mul_815: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(add_258, view_4);  view_4 = None
    sum_263: "f32[1, 128, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3], True);  mul_814 = None
    view_428: "f32[128]" = torch.ops.aten.reshape.default(sum_263, [128]);  sum_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:160, code: x = x.permute(0, 3, 1, 2)
    permute_517: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(mul_815, [0, 2, 3, 1]);  mul_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_429: "f32[25088, 128]" = torch.ops.aten.reshape.default(permute_517, [25088, 128]);  permute_517 = None
    mm_142: "f32[25088, 512]" = torch.ops.aten.mm.default(view_429, permute_518);  permute_518 = None
    permute_519: "f32[128, 25088]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_143: "f32[128, 512]" = torch.ops.aten.mm.default(permute_519, view_2);  permute_519 = view_2 = None
    permute_520: "f32[512, 128]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_264: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[128]" = torch.ops.aten.reshape.default(sum_264, [128]);  sum_264 = None
    permute_521: "f32[128, 512]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    view_431: "f32[8, 56, 56, 512]" = torch.ops.aten.reshape.default(mm_142, [8, 56, 56, 512]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:145, code: return F.gelu(input)
    mul_817: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
    mul_818: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, view_1)
    mul_819: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(mul_818, -0.5);  mul_818 = None
    exp_35: "f32[8, 56, 56, 512]" = torch.ops.aten.exp.default(mul_819);  mul_819 = None
    mul_820: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
    mul_821: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_1, mul_820);  view_1 = mul_820 = None
    add_260: "f32[8, 56, 56, 512]" = torch.ops.aten.add.Tensor(mul_817, mul_821);  mul_817 = mul_821 = None
    mul_822: "f32[8, 56, 56, 512]" = torch.ops.aten.mul.Tensor(view_431, add_260);  view_431 = add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[25088, 512]" = torch.ops.aten.reshape.default(mul_822, [25088, 512]);  mul_822 = None
    mm_144: "f32[25088, 128]" = torch.ops.aten.mm.default(view_432, permute_522);  permute_522 = None
    permute_523: "f32[512, 25088]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_145: "f32[512, 128]" = torch.ops.aten.mm.default(permute_523, view);  permute_523 = view = None
    permute_524: "f32[128, 512]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_265: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_432, [0], True);  view_432 = None
    view_433: "f32[512]" = torch.ops.aten.reshape.default(sum_265, [512]);  sum_265 = None
    permute_525: "f32[512, 128]" = torch.ops.aten.permute.default(permute_524, [1, 0]);  permute_524 = None
    view_434: "f32[8, 56, 56, 128]" = torch.ops.aten.reshape.default(mm_144, [8, 56, 56, 128]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_824: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_434, primals_3);  primals_3 = None
    mul_825: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_824, 128)
    sum_266: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_824, [3], True)
    mul_826: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_824, mul_2);  mul_824 = None
    sum_267: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_826, [3], True);  mul_826 = None
    mul_827: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_2, sum_267);  sum_267 = None
    sub_159: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_825, sum_266);  mul_825 = sum_266 = None
    sub_160: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_159, mul_827);  sub_159 = mul_827 = None
    div_40: "f32[8, 56, 56, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    mul_828: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_40, sub_160);  div_40 = sub_160 = None
    mul_829: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_434, mul_2);  mul_2 = None
    sum_268: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_829, [0, 1, 2]);  mul_829 = None
    sum_269: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_434, [0, 1, 2]);  view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:157, code: x = x.permute(0, 2, 3, 1)
    permute_526: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_828, [0, 3, 1, 2]);  mul_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(permute_526, permute_1, primals_121, [128], [1, 1], [3, 3], [1, 1], False, [0, 0], 128, [True, True, True]);  permute_526 = permute_1 = primals_121 = None
    getitem_196: "f32[8, 128, 56, 56]" = convolution_backward_38[0]
    getitem_197: "f32[128, 1, 7, 7]" = convolution_backward_38[1]
    getitem_198: "f32[128]" = convolution_backward_38[2];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:152, code: x = self.conv_dw(x)
    add_261: "f32[8, 128, 56, 56]" = torch.ops.aten.add.Tensor(add_258, getitem_196);  add_258 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    permute_527: "f32[8, 56, 56, 128]" = torch.ops.aten.permute.default(add_261, [0, 2, 3, 1]);  add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_831: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_527, primals_1);  primals_1 = None
    mul_832: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_831, 128)
    sum_270: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [3], True)
    mul_833: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul_831, mul);  mul_831 = None
    sum_271: "f32[8, 56, 56, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [3], True);  mul_833 = None
    mul_834: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(mul, sum_271);  sum_271 = None
    sub_162: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(mul_832, sum_270);  mul_832 = sum_270 = None
    sub_163: "f32[8, 56, 56, 128]" = torch.ops.aten.sub.Tensor(sub_162, mul_834);  sub_162 = mul_834 = None
    mul_835: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(div_41, sub_163);  div_41 = sub_163 = None
    mul_836: "f32[8, 56, 56, 128]" = torch.ops.aten.mul.Tensor(permute_527, mul);  mul = None
    sum_272: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1, 2]);  mul_836 = None
    sum_273: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_527, [0, 1, 2]);  permute_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    permute_528: "f32[8, 128, 56, 56]" = torch.ops.aten.permute.default(mul_835, [0, 3, 1, 2]);  mul_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convnext.py:411, code: x = self.stem(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(permute_528, primals_345, primals_119, [128], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  permute_528 = primals_345 = primals_119 = None
    getitem_200: "f32[128, 3, 4, 4]" = convolution_backward_39[1]
    getitem_201: "f32[128]" = convolution_backward_39[2];  convolution_backward_39 = None
    return [sum_272, sum_273, sum_268, sum_269, view_428, sum_261, sum_262, view_421, sum_254, sum_255, view_414, sum_247, sum_248, sum_243, sum_244, view_407, sum_236, sum_237, view_400, sum_229, sum_230, view_393, sum_222, sum_223, sum_218, sum_219, view_386, sum_211, sum_212, view_379, sum_204, sum_205, view_372, sum_197, sum_198, view_365, sum_190, sum_191, view_358, sum_183, sum_184, view_351, sum_176, sum_177, view_344, sum_169, sum_170, view_337, sum_162, sum_163, view_330, sum_155, sum_156, view_323, sum_148, sum_149, view_316, sum_141, sum_142, view_309, sum_134, sum_135, view_302, sum_127, sum_128, view_295, sum_120, sum_121, view_288, sum_113, sum_114, view_281, sum_106, sum_107, view_274, sum_99, sum_100, view_267, sum_92, sum_93, view_260, sum_85, sum_86, view_253, sum_78, sum_79, view_246, sum_71, sum_72, view_239, sum_64, sum_65, view_232, sum_57, sum_58, view_225, sum_50, sum_51, view_218, sum_43, sum_44, view_211, sum_36, sum_37, view_204, sum_29, sum_30, sum_25, sum_26, view_197, sum_18, sum_19, view_190, sum_11, sum_12, view_183, sum_4, sum_5, getitem_200, getitem_201, getitem_197, getitem_198, permute_525, view_433, permute_521, view_430, getitem_194, getitem_195, permute_515, view_426, permute_511, view_423, getitem_191, getitem_192, permute_505, view_419, permute_501, view_416, getitem_188, getitem_189, getitem_185, getitem_186, permute_493, view_412, permute_489, view_409, getitem_182, getitem_183, permute_483, view_405, permute_479, view_402, getitem_179, getitem_180, permute_473, view_398, permute_469, view_395, getitem_176, getitem_177, getitem_173, getitem_174, permute_461, view_391, permute_457, view_388, getitem_170, getitem_171, permute_451, view_384, permute_447, view_381, getitem_167, getitem_168, permute_441, view_377, permute_437, view_374, getitem_164, getitem_165, permute_431, view_370, permute_427, view_367, getitem_161, getitem_162, permute_421, view_363, permute_417, view_360, getitem_158, getitem_159, permute_411, view_356, permute_407, view_353, getitem_155, getitem_156, permute_401, view_349, permute_397, view_346, getitem_152, getitem_153, permute_391, view_342, permute_387, view_339, getitem_149, getitem_150, permute_381, view_335, permute_377, view_332, getitem_146, getitem_147, permute_371, view_328, permute_367, view_325, getitem_143, getitem_144, permute_361, view_321, permute_357, view_318, getitem_140, getitem_141, permute_351, view_314, permute_347, view_311, getitem_137, getitem_138, permute_341, view_307, permute_337, view_304, getitem_134, getitem_135, permute_331, view_300, permute_327, view_297, getitem_131, getitem_132, permute_321, view_293, permute_317, view_290, getitem_128, getitem_129, permute_311, view_286, permute_307, view_283, getitem_125, getitem_126, permute_301, view_279, permute_297, view_276, getitem_122, getitem_123, permute_291, view_272, permute_287, view_269, getitem_119, getitem_120, permute_281, view_265, permute_277, view_262, getitem_116, getitem_117, permute_271, view_258, permute_267, view_255, getitem_113, getitem_114, permute_261, view_251, permute_257, view_248, getitem_110, getitem_111, permute_251, view_244, permute_247, view_241, getitem_107, getitem_108, permute_241, view_237, permute_237, view_234, getitem_104, getitem_105, permute_231, view_230, permute_227, view_227, getitem_101, getitem_102, permute_221, view_223, permute_217, view_220, getitem_98, getitem_99, permute_211, view_216, permute_207, view_213, getitem_95, getitem_96, permute_201, view_209, permute_197, view_206, getitem_92, getitem_93, getitem_89, getitem_90, permute_189, view_202, permute_185, view_199, getitem_86, getitem_87, permute_179, view_195, permute_175, view_192, getitem_83, getitem_84, permute_169, view_188, permute_165, view_185, permute_158, view_181, None]
    