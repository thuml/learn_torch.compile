from __future__ import annotations



def forward(self, primals_1: "f32[256, 3, 16, 16]", primals_3: "f32[256]", primals_7: "f32[768]", primals_10: "f32[196]", primals_13: "f32[256]", primals_17: "f32[768]", primals_20: "f32[196]", primals_23: "f32[256]", primals_27: "f32[768]", primals_30: "f32[196]", primals_33: "f32[256]", primals_37: "f32[768]", primals_40: "f32[196]", primals_43: "f32[256]", primals_47: "f32[768]", primals_50: "f32[196]", primals_53: "f32[256]", primals_57: "f32[768]", primals_60: "f32[196]", primals_63: "f32[256]", primals_67: "f32[768]", primals_70: "f32[196]", primals_73: "f32[256]", primals_77: "f32[768]", primals_80: "f32[196]", primals_83: "f32[256]", primals_87: "f32[768]", primals_90: "f32[196]", primals_93: "f32[256]", primals_97: "f32[768]", primals_100: "f32[196]", primals_103: "f32[256]", primals_107: "f32[768]", primals_110: "f32[196]", primals_113: "f32[256]", primals_117: "f32[768]", primals_120: "f32[196]", primals_123: "f32[256]", primals_127: "f32[768]", primals_130: "f32[196]", primals_133: "f32[256]", primals_137: "f32[768]", primals_140: "f32[196]", primals_143: "f32[256]", primals_147: "f32[768]", primals_150: "f32[196]", primals_153: "f32[256]", primals_157: "f32[768]", primals_160: "f32[196]", primals_163: "f32[256]", primals_167: "f32[768]", primals_170: "f32[196]", primals_173: "f32[256]", primals_177: "f32[768]", primals_180: "f32[196]", primals_183: "f32[256]", primals_187: "f32[768]", primals_190: "f32[196]", primals_193: "f32[256]", primals_197: "f32[768]", primals_200: "f32[196]", primals_203: "f32[256]", primals_207: "f32[768]", primals_210: "f32[196]", primals_213: "f32[256]", primals_217: "f32[768]", primals_220: "f32[196]", primals_223: "f32[256]", primals_227: "f32[768]", primals_230: "f32[196]", primals_233: "f32[256]", primals_237: "f32[768]", primals_240: "f32[196]", primals_243: "f32[256]", primals_247: "f32[768]", primals_250: "f32[196]", primals_253: "f32[256]", primals_257: "f32[768]", primals_260: "f32[196]", primals_263: "f32[256]", primals_267: "f32[768]", primals_270: "f32[196]", primals_273: "f32[256]", primals_277: "f32[768]", primals_280: "f32[196]", primals_283: "f32[256]", primals_287: "f32[768]", primals_290: "f32[196]", primals_293: "f32[256]", primals_297: "f32[768]", primals_300: "f32[196]", primals_303: "f32[256]", primals_307: "f32[8, 3, 224, 224]", mul: "f32[8, 196, 256]", view_1: "f32[1568, 256]", addmm: "f32[1568, 1536]", getitem_2: "f32[8, 196, 768]", mul_5: "f32[8, 196, 768]", view_3: "f32[6144, 196]", mm: "f32[6144, 196]", view_5: "f32[1568, 768]", mul_8: "f32[8, 196, 256]", view_7: "f32[1568, 256]", addmm_2: "f32[1568, 1536]", getitem_8: "f32[8, 196, 768]", mul_13: "f32[8, 196, 768]", view_9: "f32[6144, 196]", mm_1: "f32[6144, 196]", view_11: "f32[1568, 768]", mul_16: "f32[8, 196, 256]", view_13: "f32[1568, 256]", addmm_4: "f32[1568, 1536]", getitem_14: "f32[8, 196, 768]", mul_21: "f32[8, 196, 768]", view_15: "f32[6144, 196]", mm_2: "f32[6144, 196]", view_17: "f32[1568, 768]", mul_24: "f32[8, 196, 256]", view_19: "f32[1568, 256]", addmm_6: "f32[1568, 1536]", getitem_20: "f32[8, 196, 768]", mul_29: "f32[8, 196, 768]", view_21: "f32[6144, 196]", mm_3: "f32[6144, 196]", view_23: "f32[1568, 768]", mul_32: "f32[8, 196, 256]", view_25: "f32[1568, 256]", addmm_8: "f32[1568, 1536]", getitem_26: "f32[8, 196, 768]", mul_37: "f32[8, 196, 768]", view_27: "f32[6144, 196]", mm_4: "f32[6144, 196]", view_29: "f32[1568, 768]", mul_40: "f32[8, 196, 256]", view_31: "f32[1568, 256]", addmm_10: "f32[1568, 1536]", getitem_32: "f32[8, 196, 768]", mul_45: "f32[8, 196, 768]", view_33: "f32[6144, 196]", mm_5: "f32[6144, 196]", view_35: "f32[1568, 768]", mul_48: "f32[8, 196, 256]", view_37: "f32[1568, 256]", addmm_12: "f32[1568, 1536]", getitem_38: "f32[8, 196, 768]", mul_53: "f32[8, 196, 768]", view_39: "f32[6144, 196]", mm_6: "f32[6144, 196]", view_41: "f32[1568, 768]", mul_56: "f32[8, 196, 256]", view_43: "f32[1568, 256]", addmm_14: "f32[1568, 1536]", getitem_44: "f32[8, 196, 768]", mul_61: "f32[8, 196, 768]", view_45: "f32[6144, 196]", mm_7: "f32[6144, 196]", view_47: "f32[1568, 768]", mul_64: "f32[8, 196, 256]", view_49: "f32[1568, 256]", addmm_16: "f32[1568, 1536]", getitem_50: "f32[8, 196, 768]", mul_69: "f32[8, 196, 768]", view_51: "f32[6144, 196]", mm_8: "f32[6144, 196]", view_53: "f32[1568, 768]", mul_72: "f32[8, 196, 256]", view_55: "f32[1568, 256]", addmm_18: "f32[1568, 1536]", getitem_56: "f32[8, 196, 768]", mul_77: "f32[8, 196, 768]", view_57: "f32[6144, 196]", mm_9: "f32[6144, 196]", view_59: "f32[1568, 768]", mul_80: "f32[8, 196, 256]", view_61: "f32[1568, 256]", addmm_20: "f32[1568, 1536]", getitem_62: "f32[8, 196, 768]", mul_85: "f32[8, 196, 768]", view_63: "f32[6144, 196]", mm_10: "f32[6144, 196]", view_65: "f32[1568, 768]", mul_88: "f32[8, 196, 256]", view_67: "f32[1568, 256]", addmm_22: "f32[1568, 1536]", getitem_68: "f32[8, 196, 768]", mul_93: "f32[8, 196, 768]", view_69: "f32[6144, 196]", mm_11: "f32[6144, 196]", view_71: "f32[1568, 768]", mul_96: "f32[8, 196, 256]", view_73: "f32[1568, 256]", addmm_24: "f32[1568, 1536]", getitem_74: "f32[8, 196, 768]", mul_101: "f32[8, 196, 768]", view_75: "f32[6144, 196]", mm_12: "f32[6144, 196]", view_77: "f32[1568, 768]", mul_104: "f32[8, 196, 256]", view_79: "f32[1568, 256]", addmm_26: "f32[1568, 1536]", getitem_80: "f32[8, 196, 768]", mul_109: "f32[8, 196, 768]", view_81: "f32[6144, 196]", mm_13: "f32[6144, 196]", view_83: "f32[1568, 768]", mul_112: "f32[8, 196, 256]", view_85: "f32[1568, 256]", addmm_28: "f32[1568, 1536]", getitem_86: "f32[8, 196, 768]", mul_117: "f32[8, 196, 768]", view_87: "f32[6144, 196]", mm_14: "f32[6144, 196]", view_89: "f32[1568, 768]", mul_120: "f32[8, 196, 256]", view_91: "f32[1568, 256]", addmm_30: "f32[1568, 1536]", getitem_92: "f32[8, 196, 768]", mul_125: "f32[8, 196, 768]", view_93: "f32[6144, 196]", mm_15: "f32[6144, 196]", view_95: "f32[1568, 768]", mul_128: "f32[8, 196, 256]", view_97: "f32[1568, 256]", addmm_32: "f32[1568, 1536]", getitem_98: "f32[8, 196, 768]", mul_133: "f32[8, 196, 768]", view_99: "f32[6144, 196]", mm_16: "f32[6144, 196]", view_101: "f32[1568, 768]", mul_136: "f32[8, 196, 256]", view_103: "f32[1568, 256]", addmm_34: "f32[1568, 1536]", getitem_104: "f32[8, 196, 768]", mul_141: "f32[8, 196, 768]", view_105: "f32[6144, 196]", mm_17: "f32[6144, 196]", view_107: "f32[1568, 768]", mul_144: "f32[8, 196, 256]", view_109: "f32[1568, 256]", addmm_36: "f32[1568, 1536]", getitem_110: "f32[8, 196, 768]", mul_149: "f32[8, 196, 768]", view_111: "f32[6144, 196]", mm_18: "f32[6144, 196]", view_113: "f32[1568, 768]", mul_152: "f32[8, 196, 256]", view_115: "f32[1568, 256]", addmm_38: "f32[1568, 1536]", getitem_116: "f32[8, 196, 768]", mul_157: "f32[8, 196, 768]", view_117: "f32[6144, 196]", mm_19: "f32[6144, 196]", view_119: "f32[1568, 768]", mul_160: "f32[8, 196, 256]", view_121: "f32[1568, 256]", addmm_40: "f32[1568, 1536]", getitem_122: "f32[8, 196, 768]", mul_165: "f32[8, 196, 768]", view_123: "f32[6144, 196]", mm_20: "f32[6144, 196]", view_125: "f32[1568, 768]", mul_168: "f32[8, 196, 256]", view_127: "f32[1568, 256]", addmm_42: "f32[1568, 1536]", getitem_128: "f32[8, 196, 768]", mul_173: "f32[8, 196, 768]", view_129: "f32[6144, 196]", mm_21: "f32[6144, 196]", view_131: "f32[1568, 768]", mul_176: "f32[8, 196, 256]", view_133: "f32[1568, 256]", addmm_44: "f32[1568, 1536]", getitem_134: "f32[8, 196, 768]", mul_181: "f32[8, 196, 768]", view_135: "f32[6144, 196]", mm_22: "f32[6144, 196]", view_137: "f32[1568, 768]", mul_184: "f32[8, 196, 256]", view_139: "f32[1568, 256]", addmm_46: "f32[1568, 1536]", getitem_140: "f32[8, 196, 768]", mul_189: "f32[8, 196, 768]", view_141: "f32[6144, 196]", mm_23: "f32[6144, 196]", view_143: "f32[1568, 768]", mul_192: "f32[8, 196, 256]", view_145: "f32[1568, 256]", addmm_48: "f32[1568, 1536]", getitem_146: "f32[8, 196, 768]", mul_197: "f32[8, 196, 768]", view_147: "f32[6144, 196]", mm_24: "f32[6144, 196]", view_149: "f32[1568, 768]", mul_200: "f32[8, 196, 256]", view_151: "f32[1568, 256]", addmm_50: "f32[1568, 1536]", getitem_152: "f32[8, 196, 768]", mul_205: "f32[8, 196, 768]", view_153: "f32[6144, 196]", mm_25: "f32[6144, 196]", view_155: "f32[1568, 768]", mul_208: "f32[8, 196, 256]", view_157: "f32[1568, 256]", addmm_52: "f32[1568, 1536]", getitem_158: "f32[8, 196, 768]", mul_213: "f32[8, 196, 768]", view_159: "f32[6144, 196]", mm_26: "f32[6144, 196]", view_161: "f32[1568, 768]", mul_216: "f32[8, 196, 256]", view_163: "f32[1568, 256]", addmm_54: "f32[1568, 1536]", getitem_164: "f32[8, 196, 768]", mul_221: "f32[8, 196, 768]", view_165: "f32[6144, 196]", mm_27: "f32[6144, 196]", view_167: "f32[1568, 768]", mul_224: "f32[8, 196, 256]", view_169: "f32[1568, 256]", addmm_56: "f32[1568, 1536]", getitem_170: "f32[8, 196, 768]", mul_229: "f32[8, 196, 768]", view_171: "f32[6144, 196]", mm_28: "f32[6144, 196]", view_173: "f32[1568, 768]", mul_232: "f32[8, 196, 256]", view_175: "f32[1568, 256]", addmm_58: "f32[1568, 1536]", getitem_176: "f32[8, 196, 768]", mul_237: "f32[8, 196, 768]", view_177: "f32[6144, 196]", mm_29: "f32[6144, 196]", view_179: "f32[1568, 768]", mul_240: "f32[8, 196, 256]", clone_151: "f32[8, 256]", permute_152: "f32[1000, 256]", div_1: "f32[8, 196, 1]", permute_156: "f32[256, 768]", permute_163: "f32[196, 196]", div_2: "f32[8, 196, 1]", permute_166: "f32[1536, 256]", div_3: "f32[8, 196, 1]", permute_170: "f32[256, 768]", permute_177: "f32[196, 196]", div_4: "f32[8, 196, 1]", permute_180: "f32[1536, 256]", div_5: "f32[8, 196, 1]", permute_184: "f32[256, 768]", permute_191: "f32[196, 196]", div_6: "f32[8, 196, 1]", permute_194: "f32[1536, 256]", div_7: "f32[8, 196, 1]", permute_198: "f32[256, 768]", permute_205: "f32[196, 196]", div_8: "f32[8, 196, 1]", permute_208: "f32[1536, 256]", div_9: "f32[8, 196, 1]", permute_212: "f32[256, 768]", permute_219: "f32[196, 196]", div_10: "f32[8, 196, 1]", permute_222: "f32[1536, 256]", div_11: "f32[8, 196, 1]", permute_226: "f32[256, 768]", permute_233: "f32[196, 196]", div_12: "f32[8, 196, 1]", permute_236: "f32[1536, 256]", div_13: "f32[8, 196, 1]", permute_240: "f32[256, 768]", permute_247: "f32[196, 196]", div_14: "f32[8, 196, 1]", permute_250: "f32[1536, 256]", div_15: "f32[8, 196, 1]", permute_254: "f32[256, 768]", permute_261: "f32[196, 196]", div_16: "f32[8, 196, 1]", permute_264: "f32[1536, 256]", div_17: "f32[8, 196, 1]", permute_268: "f32[256, 768]", permute_275: "f32[196, 196]", div_18: "f32[8, 196, 1]", permute_278: "f32[1536, 256]", div_19: "f32[8, 196, 1]", permute_282: "f32[256, 768]", permute_289: "f32[196, 196]", div_20: "f32[8, 196, 1]", permute_292: "f32[1536, 256]", div_21: "f32[8, 196, 1]", permute_296: "f32[256, 768]", permute_303: "f32[196, 196]", div_22: "f32[8, 196, 1]", permute_306: "f32[1536, 256]", div_23: "f32[8, 196, 1]", permute_310: "f32[256, 768]", permute_317: "f32[196, 196]", div_24: "f32[8, 196, 1]", permute_320: "f32[1536, 256]", div_25: "f32[8, 196, 1]", permute_324: "f32[256, 768]", permute_331: "f32[196, 196]", div_26: "f32[8, 196, 1]", permute_334: "f32[1536, 256]", div_27: "f32[8, 196, 1]", permute_338: "f32[256, 768]", permute_345: "f32[196, 196]", div_28: "f32[8, 196, 1]", permute_348: "f32[1536, 256]", div_29: "f32[8, 196, 1]", permute_352: "f32[256, 768]", permute_359: "f32[196, 196]", div_30: "f32[8, 196, 1]", permute_362: "f32[1536, 256]", div_31: "f32[8, 196, 1]", permute_366: "f32[256, 768]", permute_373: "f32[196, 196]", div_32: "f32[8, 196, 1]", permute_376: "f32[1536, 256]", div_33: "f32[8, 196, 1]", permute_380: "f32[256, 768]", permute_387: "f32[196, 196]", div_34: "f32[8, 196, 1]", permute_390: "f32[1536, 256]", div_35: "f32[8, 196, 1]", permute_394: "f32[256, 768]", permute_401: "f32[196, 196]", div_36: "f32[8, 196, 1]", permute_404: "f32[1536, 256]", div_37: "f32[8, 196, 1]", permute_408: "f32[256, 768]", permute_415: "f32[196, 196]", div_38: "f32[8, 196, 1]", permute_418: "f32[1536, 256]", div_39: "f32[8, 196, 1]", permute_422: "f32[256, 768]", permute_429: "f32[196, 196]", div_40: "f32[8, 196, 1]", permute_432: "f32[1536, 256]", div_41: "f32[8, 196, 1]", permute_436: "f32[256, 768]", permute_443: "f32[196, 196]", div_42: "f32[8, 196, 1]", permute_446: "f32[1536, 256]", div_43: "f32[8, 196, 1]", permute_450: "f32[256, 768]", permute_457: "f32[196, 196]", div_44: "f32[8, 196, 1]", permute_460: "f32[1536, 256]", div_45: "f32[8, 196, 1]", permute_464: "f32[256, 768]", permute_471: "f32[196, 196]", div_46: "f32[8, 196, 1]", permute_474: "f32[1536, 256]", div_47: "f32[8, 196, 1]", permute_478: "f32[256, 768]", permute_485: "f32[196, 196]", div_48: "f32[8, 196, 1]", permute_488: "f32[1536, 256]", div_49: "f32[8, 196, 1]", permute_492: "f32[256, 768]", permute_499: "f32[196, 196]", div_50: "f32[8, 196, 1]", permute_502: "f32[1536, 256]", div_51: "f32[8, 196, 1]", permute_506: "f32[256, 768]", permute_513: "f32[196, 196]", div_52: "f32[8, 196, 1]", permute_516: "f32[1536, 256]", div_53: "f32[8, 196, 1]", permute_520: "f32[256, 768]", permute_527: "f32[196, 196]", div_54: "f32[8, 196, 1]", permute_530: "f32[1536, 256]", div_55: "f32[8, 196, 1]", permute_534: "f32[256, 768]", permute_541: "f32[196, 196]", div_56: "f32[8, 196, 1]", permute_544: "f32[1536, 256]", div_57: "f32[8, 196, 1]", permute_548: "f32[256, 768]", permute_555: "f32[196, 196]", div_58: "f32[8, 196, 1]", permute_558: "f32[1536, 256]", div_59: "f32[8, 196, 1]", permute_562: "f32[256, 768]", permute_569: "f32[196, 196]", div_60: "f32[8, 196, 1]", permute_572: "f32[1536, 256]", div_61: "f32[8, 196, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_2: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm, [8, 196, 1536]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_3: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, 0.7071067811865476)
    erf: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_2: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_4: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm, [8, 768, 196]);  mm = None
    add_5: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_4, primals_10);  view_4 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_4: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_5, [0, 2, 1]);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_8: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_2, [8, 196, 1536]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_11: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf_1: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
    add_9: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_10: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_1, [8, 768, 196]);  mm_1 = None
    add_12: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_10, primals_20);  view_10 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_9: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_12, [0, 2, 1]);  add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_14: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_4, [8, 196, 1536]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_19: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
    erf_2: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_16: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_16: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_2, [8, 768, 196]);  mm_2 = None
    add_19: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_16, primals_30);  view_16 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_14: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_19, [0, 2, 1]);  add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_20: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_6, [8, 196, 1536]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_27: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, 0.7071067811865476)
    erf_3: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_23: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_22: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_3, [8, 768, 196]);  mm_3 = None
    add_26: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_22, primals_40);  view_22 = primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_19: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_26, [0, 2, 1]);  add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_26: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_8, [8, 196, 1536]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_35: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, 0.7071067811865476)
    erf_4: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_30: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_28: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_4, [8, 768, 196]);  mm_4 = None
    add_33: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_28, primals_50);  view_28 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_24: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_33, [0, 2, 1]);  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_32: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_10, [8, 196, 1536]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_43: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, 0.7071067811865476)
    erf_5: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_37: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_34: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_5, [8, 768, 196]);  mm_5 = None
    add_40: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_34, primals_60);  view_34 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_29: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_40, [0, 2, 1]);  add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_38: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 196, 1536]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_51: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_6: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_44: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_40: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_6, [8, 768, 196]);  mm_6 = None
    add_47: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_40, primals_70);  view_40 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_34: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_47, [0, 2, 1]);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_44: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_14, [8, 196, 1536]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_59: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476)
    erf_7: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_59);  mul_59 = None
    add_51: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_46: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_7, [8, 768, 196]);  mm_7 = None
    add_54: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_46, primals_80);  view_46 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_39: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_54, [0, 2, 1]);  add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_50: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 196, 1536]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_67: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_8: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_67);  mul_67 = None
    add_58: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_52: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_8, [8, 768, 196]);  mm_8 = None
    add_61: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_52, primals_90);  view_52 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_44: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_61, [0, 2, 1]);  add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_56: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_18, [8, 196, 1536]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_75: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, 0.7071067811865476)
    erf_9: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_65: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_58: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_9, [8, 768, 196]);  mm_9 = None
    add_68: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_58, primals_100);  view_58 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_49: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_68, [0, 2, 1]);  add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_62: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 196, 1536]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_83: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476)
    erf_10: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_72: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_64: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_10, [8, 768, 196]);  mm_10 = None
    add_75: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_64, primals_110);  view_64 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_54: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_75, [0, 2, 1]);  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_68: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_22, [8, 196, 1536]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_91: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_11: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_91);  mul_91 = None
    add_79: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_70: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_11, [8, 768, 196]);  mm_11 = None
    add_82: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_70, primals_120);  view_70 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_59: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_82, [0, 2, 1]);  add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_74: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 196, 1536]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_99: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, 0.7071067811865476)
    erf_12: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_86: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_76: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_12, [8, 768, 196]);  mm_12 = None
    add_89: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_76, primals_130);  view_76 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_64: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_89, [0, 2, 1]);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_80: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_26, [8, 196, 1536]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_107: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_13: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_93: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_82: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_13, [8, 768, 196]);  mm_13 = None
    add_96: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_82, primals_140);  view_82 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_69: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_96, [0, 2, 1]);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_86: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 196, 1536]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_115: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_14: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_115);  mul_115 = None
    add_100: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_88: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_14, [8, 768, 196]);  mm_14 = None
    add_103: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_88, primals_150);  view_88 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_74: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_103, [0, 2, 1]);  add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_92: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_30, [8, 196, 1536]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_123: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476)
    erf_15: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_107: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_94: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_15, [8, 768, 196]);  mm_15 = None
    add_110: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_94, primals_160);  view_94 = primals_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_79: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_110, [0, 2, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_98: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 196, 1536]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_131: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_16: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
    add_114: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_100: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_16, [8, 768, 196]);  mm_16 = None
    add_117: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_100, primals_170);  view_100 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_84: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_117, [0, 2, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_104: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_34, [8, 196, 1536]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_139: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476)
    erf_17: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_139);  mul_139 = None
    add_121: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_106: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_17, [8, 768, 196]);  mm_17 = None
    add_124: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_106, primals_180);  view_106 = primals_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_89: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_124, [0, 2, 1]);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_110: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_36, [8, 196, 1536]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_147: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, 0.7071067811865476)
    erf_18: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_147);  mul_147 = None
    add_128: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_112: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_18, [8, 768, 196]);  mm_18 = None
    add_131: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_112, primals_190);  view_112 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_94: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_131, [0, 2, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_116: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_38, [8, 196, 1536]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_155: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_19: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_135: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_118: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_19, [8, 768, 196]);  mm_19 = None
    add_138: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_118, primals_200);  view_118 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_99: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_138, [0, 2, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_122: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_40, [8, 196, 1536]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_163: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
    erf_20: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
    add_142: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_124: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_20, [8, 768, 196]);  mm_20 = None
    add_145: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_124, primals_210);  view_124 = primals_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_104: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_145, [0, 2, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_128: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_42, [8, 196, 1536]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_171: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_21: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_171);  mul_171 = None
    add_149: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_130: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_21, [8, 768, 196]);  mm_21 = None
    add_152: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_130, primals_220);  view_130 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_109: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_152, [0, 2, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_134: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_44, [8, 196, 1536]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_179: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, 0.7071067811865476)
    erf_22: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_179);  mul_179 = None
    add_156: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_136: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_22, [8, 768, 196]);  mm_22 = None
    add_159: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_136, primals_230);  view_136 = primals_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_114: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_159, [0, 2, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_140: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_46, [8, 196, 1536]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_187: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, 0.7071067811865476)
    erf_23: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
    add_163: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_142: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_23, [8, 768, 196]);  mm_23 = None
    add_166: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_142, primals_240);  view_142 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_119: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_166, [0, 2, 1]);  add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_146: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_48, [8, 196, 1536]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_195: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, 0.7071067811865476)
    erf_24: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_170: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_148: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_24, [8, 768, 196]);  mm_24 = None
    add_173: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_148, primals_250);  view_148 = primals_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_124: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_152: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_50, [8, 196, 1536]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_203: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, 0.7071067811865476)
    erf_25: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_203);  mul_203 = None
    add_177: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_154: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_25, [8, 768, 196]);  mm_25 = None
    add_180: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_154, primals_260);  view_154 = primals_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_129: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_180, [0, 2, 1]);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_158: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_52, [8, 196, 1536]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_211: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476)
    erf_26: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_211);  mul_211 = None
    add_184: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_160: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_26, [8, 768, 196]);  mm_26 = None
    add_187: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_160, primals_270);  view_160 = primals_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_134: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_187, [0, 2, 1]);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_164: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_54, [8, 196, 1536]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_219: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
    erf_27: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_191: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_166: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_27, [8, 768, 196]);  mm_27 = None
    add_194: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_166, primals_280);  view_166 = primals_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_139: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_194, [0, 2, 1]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_170: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_56, [8, 196, 1536]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_227: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, 0.7071067811865476)
    erf_28: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_227);  mul_227 = None
    add_198: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_172: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_28, [8, 768, 196]);  mm_28 = None
    add_201: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_172, primals_290);  view_172 = primals_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_144: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_201, [0, 2, 1]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_176: "f32[8, 196, 1536]" = torch.ops.aten.view.default(addmm_58, [8, 196, 1536]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_235: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, 0.7071067811865476)
    erf_29: "f32[8, 196, 1536]" = torch.ops.aten.erf.default(mul_235);  mul_235 = None
    add_205: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    view_178: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_29, [8, 768, 196]);  mm_29 = None
    add_208: "f32[8, 768, 196]" = torch.ops.aten.add.Tensor(view_178, primals_300);  view_178 = primals_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    permute_149: "f32[8, 196, 768]" = torch.ops.aten.permute.default(add_208, [0, 2, 1]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    mm_30: "f32[8, 256]" = torch.ops.aten.mm.default(tangents_1, permute_152);  permute_152 = None
    permute_153: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_31: "f32[1000, 256]" = torch.ops.aten.mm.default(permute_153, clone_151);  permute_153 = clone_151 = None
    permute_154: "f32[256, 1000]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_181: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_155: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 256]" = torch.ops.aten.unsqueeze.default(mm_30, 1);  mm_30 = None
    expand: "f32[8, 196, 256]" = torch.ops.aten.expand.default(unsqueeze, [8, 196, 256]);  unsqueeze = None
    div: "f32[8, 196, 256]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    mul_243: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div, primals_303);  primals_303 = None
    mul_244: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_243, 256)
    sum_2: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
    mul_245: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_243, mul_240);  mul_243 = None
    sum_3: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
    mul_246: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_240, sum_3);  sum_3 = None
    sub_62: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_244, sum_2);  mul_244 = sum_2 = None
    sub_63: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_62, mul_246);  sub_62 = mul_246 = None
    mul_247: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_1, sub_63);  div_1 = sub_63 = None
    mul_248: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div, mul_240);  mul_240 = None
    sum_4: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
    sum_5: "f32[256]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_182: "f32[1568, 256]" = torch.ops.aten.view.default(mul_247, [1568, 256])
    mm_32: "f32[1568, 768]" = torch.ops.aten.mm.default(view_182, permute_156);  permute_156 = None
    permute_157: "f32[256, 1568]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_33: "f32[256, 768]" = torch.ops.aten.mm.default(permute_157, view_179);  permute_157 = view_179 = None
    permute_158: "f32[768, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_6: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[256]" = torch.ops.aten.view.default(sum_6, [256]);  sum_6 = None
    permute_159: "f32[256, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    view_184: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_32, [8, 196, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_184, getitem_176);  getitem_176 = None
    mul_250: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_184, permute_149);  view_184 = permute_149 = None
    permute_160: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_249, [0, 2, 1]);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_7: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_160, [0, 1], True)
    view_185: "f32[196]" = torch.ops.aten.view.default(sum_7, [196]);  sum_7 = None
    clone_153: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_186: "f32[6144, 196]" = torch.ops.aten.view.default(clone_153, [6144, 196]);  clone_153 = None
    permute_161: "f32[196, 6144]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_34: "f32[196, 196]" = torch.ops.aten.mm.default(permute_161, view_177);  permute_161 = view_177 = None
    permute_162: "f32[196, 196]" = torch.ops.aten.permute.default(mm_34, [1, 0]);  mm_34 = None
    mm_35: "f32[6144, 196]" = torch.ops.aten.mm.default(view_186, permute_163);  view_186 = permute_163 = None
    view_187: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_35, [8, 768, 196]);  mm_35 = None
    permute_164: "f32[196, 196]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
    permute_165: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_154: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_154, primals_297);  primals_297 = None
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_252, 768)
    sum_8: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
    mul_254: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_237);  mul_252 = None
    sum_9: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
    mul_255: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, sum_9);  sum_9 = None
    sub_65: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_253, sum_8);  mul_253 = sum_8 = None
    sub_66: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_255);  sub_65 = mul_255 = None
    mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_66);  div_2 = sub_66 = None
    mul_257: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_154, mul_237);  mul_237 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_154, [0, 1]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_250, mul_256], 2);  mul_250 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_259: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_205, 0.5);  add_205 = None
    mul_260: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, view_176)
    mul_261: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_260, -0.5);  mul_260 = None
    exp: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_261);  mul_261 = None
    mul_262: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_263: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_176, mul_262);  view_176 = mul_262 = None
    add_213: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_259, mul_263);  mul_259 = mul_263 = None
    mul_264: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat, add_213);  cat = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_188: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_264, [1568, 1536]);  mul_264 = None
    mm_36: "f32[1568, 256]" = torch.ops.aten.mm.default(view_188, permute_166);  permute_166 = None
    permute_167: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_37: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_167, view_175);  permute_167 = view_175 = None
    permute_168: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_12: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[1536]" = torch.ops.aten.view.default(sum_12, [1536]);  sum_12 = None
    permute_169: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_190: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_36, [8, 196, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_266: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_190, primals_293);  primals_293 = None
    mul_267: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_266, 256)
    sum_13: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_266, mul_232);  mul_266 = None
    sum_14: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_232, sum_14);  sum_14 = None
    sub_68: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_267, sum_13);  mul_267 = sum_13 = None
    sub_69: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_68, mul_269);  sub_68 = mul_269 = None
    mul_270: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_3, sub_69);  div_3 = sub_69 = None
    mul_271: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_190, mul_232);  mul_232 = None
    sum_15: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_16: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_190, [0, 1]);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_214: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(mul_247, mul_270);  mul_247 = mul_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_191: "f32[1568, 256]" = torch.ops.aten.view.default(add_214, [1568, 256])
    mm_38: "f32[1568, 768]" = torch.ops.aten.mm.default(view_191, permute_170);  permute_170 = None
    permute_171: "f32[256, 1568]" = torch.ops.aten.permute.default(view_191, [1, 0])
    mm_39: "f32[256, 768]" = torch.ops.aten.mm.default(permute_171, view_173);  permute_171 = view_173 = None
    permute_172: "f32[768, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_17: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True);  view_191 = None
    view_192: "f32[256]" = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
    permute_173: "f32[256, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_193: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_38, [8, 196, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_193, getitem_170);  getitem_170 = None
    mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_193, permute_144);  view_193 = permute_144 = None
    permute_174: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_272, [0, 2, 1]);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_18: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_174, [0, 1], True)
    view_194: "f32[196]" = torch.ops.aten.view.default(sum_18, [196]);  sum_18 = None
    clone_157: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_174, memory_format = torch.contiguous_format);  permute_174 = None
    view_195: "f32[6144, 196]" = torch.ops.aten.view.default(clone_157, [6144, 196]);  clone_157 = None
    permute_175: "f32[196, 6144]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_40: "f32[196, 196]" = torch.ops.aten.mm.default(permute_175, view_171);  permute_175 = view_171 = None
    permute_176: "f32[196, 196]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    mm_41: "f32[6144, 196]" = torch.ops.aten.mm.default(view_195, permute_177);  view_195 = permute_177 = None
    view_196: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_41, [8, 768, 196]);  mm_41 = None
    permute_178: "f32[196, 196]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    permute_179: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_196, [0, 2, 1]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_158: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    mul_275: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_158, primals_287);  primals_287 = None
    mul_276: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_275, 768)
    sum_19: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [2], True)
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_229);  mul_275 = None
    sum_20: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
    mul_278: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_229, sum_20);  sum_20 = None
    sub_71: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_276, sum_19);  mul_276 = sum_19 = None
    sub_72: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_278);  sub_71 = mul_278 = None
    mul_279: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_72);  div_4 = sub_72 = None
    mul_280: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_158, mul_229);  mul_229 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 1]);  mul_280 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_158, [0, 1]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_1: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_273, mul_279], 2);  mul_273 = mul_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_282: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_198, 0.5);  add_198 = None
    mul_283: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, view_170)
    mul_284: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_283, -0.5);  mul_283 = None
    exp_1: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_284);  mul_284 = None
    mul_285: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_286: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_170, mul_285);  view_170 = mul_285 = None
    add_216: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_282, mul_286);  mul_282 = mul_286 = None
    mul_287: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_1, add_216);  cat_1 = add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_197: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_287, [1568, 1536]);  mul_287 = None
    mm_42: "f32[1568, 256]" = torch.ops.aten.mm.default(view_197, permute_180);  permute_180 = None
    permute_181: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_43: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_181, view_169);  permute_181 = view_169 = None
    permute_182: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_23: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[1536]" = torch.ops.aten.view.default(sum_23, [1536]);  sum_23 = None
    permute_183: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_199: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_42, [8, 196, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_289: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_199, primals_283);  primals_283 = None
    mul_290: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_289, 256)
    sum_24: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True)
    mul_291: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_289, mul_224);  mul_289 = None
    sum_25: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
    mul_292: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_224, sum_25);  sum_25 = None
    sub_74: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_290, sum_24);  mul_290 = sum_24 = None
    sub_75: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_74, mul_292);  sub_74 = mul_292 = None
    mul_293: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_5, sub_75);  div_5 = sub_75 = None
    mul_294: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_199, mul_224);  mul_224 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 1]);  mul_294 = None
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_217: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_214, mul_293);  add_214 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_200: "f32[1568, 256]" = torch.ops.aten.view.default(add_217, [1568, 256])
    mm_44: "f32[1568, 768]" = torch.ops.aten.mm.default(view_200, permute_184);  permute_184 = None
    permute_185: "f32[256, 1568]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_45: "f32[256, 768]" = torch.ops.aten.mm.default(permute_185, view_167);  permute_185 = view_167 = None
    permute_186: "f32[768, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_28: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[256]" = torch.ops.aten.view.default(sum_28, [256]);  sum_28 = None
    permute_187: "f32[256, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_202: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_44, [8, 196, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_295: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_202, getitem_164);  getitem_164 = None
    mul_296: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_202, permute_139);  view_202 = permute_139 = None
    permute_188: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_295, [0, 2, 1]);  mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_29: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_188, [0, 1], True)
    view_203: "f32[196]" = torch.ops.aten.view.default(sum_29, [196]);  sum_29 = None
    clone_161: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_204: "f32[6144, 196]" = torch.ops.aten.view.default(clone_161, [6144, 196]);  clone_161 = None
    permute_189: "f32[196, 6144]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_46: "f32[196, 196]" = torch.ops.aten.mm.default(permute_189, view_165);  permute_189 = view_165 = None
    permute_190: "f32[196, 196]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    mm_47: "f32[6144, 196]" = torch.ops.aten.mm.default(view_204, permute_191);  view_204 = permute_191 = None
    view_205: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_47, [8, 768, 196]);  mm_47 = None
    permute_192: "f32[196, 196]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    permute_193: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_162: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    mul_298: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_162, primals_277);  primals_277 = None
    mul_299: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_298, 768)
    sum_30: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
    mul_300: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_298, mul_221);  mul_298 = None
    sum_31: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
    mul_301: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, sum_31);  sum_31 = None
    sub_77: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_299, sum_30);  mul_299 = sum_30 = None
    sub_78: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_301);  sub_77 = mul_301 = None
    mul_302: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_78);  div_6 = sub_78 = None
    mul_303: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_162, mul_221);  mul_221 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_162, [0, 1]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_2: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_296, mul_302], 2);  mul_296 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_305: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_191, 0.5);  add_191 = None
    mul_306: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, view_164)
    mul_307: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_306, -0.5);  mul_306 = None
    exp_2: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_307);  mul_307 = None
    mul_308: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_309: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_164, mul_308);  view_164 = mul_308 = None
    add_219: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_305, mul_309);  mul_305 = mul_309 = None
    mul_310: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_2, add_219);  cat_2 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_206: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_310, [1568, 1536]);  mul_310 = None
    mm_48: "f32[1568, 256]" = torch.ops.aten.mm.default(view_206, permute_194);  permute_194 = None
    permute_195: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_206, [1, 0])
    mm_49: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_195, view_163);  permute_195 = view_163 = None
    permute_196: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_34: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
    view_207: "f32[1536]" = torch.ops.aten.view.default(sum_34, [1536]);  sum_34 = None
    permute_197: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_208: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_48, [8, 196, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_312: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_208, primals_273);  primals_273 = None
    mul_313: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_312, 256)
    sum_35: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_312, mul_216);  mul_312 = None
    sum_36: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_216, sum_36);  sum_36 = None
    sub_80: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_313, sum_35);  mul_313 = sum_35 = None
    sub_81: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_80, mul_315);  sub_80 = mul_315 = None
    mul_316: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_7, sub_81);  div_7 = sub_81 = None
    mul_317: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_208, mul_216);  mul_216 = None
    sum_37: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_208, [0, 1]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_220: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_217, mul_316);  add_217 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_209: "f32[1568, 256]" = torch.ops.aten.view.default(add_220, [1568, 256])
    mm_50: "f32[1568, 768]" = torch.ops.aten.mm.default(view_209, permute_198);  permute_198 = None
    permute_199: "f32[256, 1568]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_51: "f32[256, 768]" = torch.ops.aten.mm.default(permute_199, view_161);  permute_199 = view_161 = None
    permute_200: "f32[768, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_39: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[256]" = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
    permute_201: "f32[256, 768]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_211: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_50, [8, 196, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_318: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, getitem_158);  getitem_158 = None
    mul_319: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, permute_134);  view_211 = permute_134 = None
    permute_202: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_318, [0, 2, 1]);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_40: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_202, [0, 1], True)
    view_212: "f32[196]" = torch.ops.aten.view.default(sum_40, [196]);  sum_40 = None
    clone_165: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    view_213: "f32[6144, 196]" = torch.ops.aten.view.default(clone_165, [6144, 196]);  clone_165 = None
    permute_203: "f32[196, 6144]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_52: "f32[196, 196]" = torch.ops.aten.mm.default(permute_203, view_159);  permute_203 = view_159 = None
    permute_204: "f32[196, 196]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    mm_53: "f32[6144, 196]" = torch.ops.aten.mm.default(view_213, permute_205);  view_213 = permute_205 = None
    view_214: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_53, [8, 768, 196]);  mm_53 = None
    permute_206: "f32[196, 196]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    permute_207: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_214, [0, 2, 1]);  view_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_166: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_166, primals_267);  primals_267 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, 768)
    sum_41: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, mul_213);  mul_321 = None
    sum_42: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_213, sum_42);  sum_42 = None
    sub_83: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_322, sum_41);  mul_322 = sum_41 = None
    sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_324);  sub_83 = mul_324 = None
    mul_325: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_84);  div_8 = sub_84 = None
    mul_326: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_166, mul_213);  mul_213 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
    sum_44: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_166, [0, 1]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_3: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_319, mul_325], 2);  mul_319 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_328: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_329: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, view_158)
    mul_330: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_329, -0.5);  mul_329 = None
    exp_3: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_330);  mul_330 = None
    mul_331: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_332: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_158, mul_331);  view_158 = mul_331 = None
    add_222: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_328, mul_332);  mul_328 = mul_332 = None
    mul_333: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_3, add_222);  cat_3 = add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_215: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_333, [1568, 1536]);  mul_333 = None
    mm_54: "f32[1568, 256]" = torch.ops.aten.mm.default(view_215, permute_208);  permute_208 = None
    permute_209: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_55: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_209, view_157);  permute_209 = view_157 = None
    permute_210: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_45: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[1536]" = torch.ops.aten.view.default(sum_45, [1536]);  sum_45 = None
    permute_211: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_217: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_54, [8, 196, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_335: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_217, primals_263);  primals_263 = None
    mul_336: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_335, 256)
    sum_46: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
    mul_337: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_335, mul_208);  mul_335 = None
    sum_47: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
    mul_338: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_208, sum_47);  sum_47 = None
    sub_86: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_336, sum_46);  mul_336 = sum_46 = None
    sub_87: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_86, mul_338);  sub_86 = mul_338 = None
    mul_339: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_9, sub_87);  div_9 = sub_87 = None
    mul_340: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_217, mul_208);  mul_208 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_223: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_220, mul_339);  add_220 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_218: "f32[1568, 256]" = torch.ops.aten.view.default(add_223, [1568, 256])
    mm_56: "f32[1568, 768]" = torch.ops.aten.mm.default(view_218, permute_212);  permute_212 = None
    permute_213: "f32[256, 1568]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_57: "f32[256, 768]" = torch.ops.aten.mm.default(permute_213, view_155);  permute_213 = view_155 = None
    permute_214: "f32[768, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_215: "f32[256, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_220: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_56, [8, 196, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_341: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, getitem_152);  getitem_152 = None
    mul_342: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_220, permute_129);  view_220 = permute_129 = None
    permute_216: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_341, [0, 2, 1]);  mul_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_51: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_216, [0, 1], True)
    view_221: "f32[196]" = torch.ops.aten.view.default(sum_51, [196]);  sum_51 = None
    clone_169: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_222: "f32[6144, 196]" = torch.ops.aten.view.default(clone_169, [6144, 196]);  clone_169 = None
    permute_217: "f32[196, 6144]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_58: "f32[196, 196]" = torch.ops.aten.mm.default(permute_217, view_153);  permute_217 = view_153 = None
    permute_218: "f32[196, 196]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    mm_59: "f32[6144, 196]" = torch.ops.aten.mm.default(view_222, permute_219);  view_222 = permute_219 = None
    view_223: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_59, [8, 768, 196]);  mm_59 = None
    permute_220: "f32[196, 196]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    permute_221: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_223, [0, 2, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_170: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_170, primals_257);  primals_257 = None
    mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_344, 768)
    sum_52: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True)
    mul_346: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_344, mul_205);  mul_344 = None
    sum_53: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True);  mul_346 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_205, sum_53);  sum_53 = None
    sub_89: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_345, sum_52);  mul_345 = sum_52 = None
    sub_90: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_347);  sub_89 = mul_347 = None
    mul_348: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_90);  div_10 = sub_90 = None
    mul_349: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_170, mul_205);  mul_205 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1]);  mul_349 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_170, [0, 1]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_4: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_342, mul_348], 2);  mul_342 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_351: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_177, 0.5);  add_177 = None
    mul_352: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, view_152)
    mul_353: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_352, -0.5);  mul_352 = None
    exp_4: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_353);  mul_353 = None
    mul_354: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_355: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_152, mul_354);  view_152 = mul_354 = None
    add_225: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_351, mul_355);  mul_351 = mul_355 = None
    mul_356: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_4, add_225);  cat_4 = add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_224: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_356, [1568, 1536]);  mul_356 = None
    mm_60: "f32[1568, 256]" = torch.ops.aten.mm.default(view_224, permute_222);  permute_222 = None
    permute_223: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_61: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_223, view_151);  permute_223 = view_151 = None
    permute_224: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_56: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[1536]" = torch.ops.aten.view.default(sum_56, [1536]);  sum_56 = None
    permute_225: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_226: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_60, [8, 196, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_358: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_226, primals_253);  primals_253 = None
    mul_359: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_358, 256)
    sum_57: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
    mul_360: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_358, mul_200);  mul_358 = None
    sum_58: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
    mul_361: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_200, sum_58);  sum_58 = None
    sub_92: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_359, sum_57);  mul_359 = sum_57 = None
    sub_93: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_92, mul_361);  sub_92 = mul_361 = None
    mul_362: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_11, sub_93);  div_11 = sub_93 = None
    mul_363: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_226, mul_200);  mul_200 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_226: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_223, mul_362);  add_223 = mul_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_227: "f32[1568, 256]" = torch.ops.aten.view.default(add_226, [1568, 256])
    mm_62: "f32[1568, 768]" = torch.ops.aten.mm.default(view_227, permute_226);  permute_226 = None
    permute_227: "f32[256, 1568]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_63: "f32[256, 768]" = torch.ops.aten.mm.default(permute_227, view_149);  permute_227 = view_149 = None
    permute_228: "f32[768, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_61: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[256]" = torch.ops.aten.view.default(sum_61, [256]);  sum_61 = None
    permute_229: "f32[256, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_229: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_62, [8, 196, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_364: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_229, getitem_146);  getitem_146 = None
    mul_365: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_229, permute_124);  view_229 = permute_124 = None
    permute_230: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_364, [0, 2, 1]);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_62: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_230, [0, 1], True)
    view_230: "f32[196]" = torch.ops.aten.view.default(sum_62, [196]);  sum_62 = None
    clone_173: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_231: "f32[6144, 196]" = torch.ops.aten.view.default(clone_173, [6144, 196]);  clone_173 = None
    permute_231: "f32[196, 6144]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_64: "f32[196, 196]" = torch.ops.aten.mm.default(permute_231, view_147);  permute_231 = view_147 = None
    permute_232: "f32[196, 196]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    mm_65: "f32[6144, 196]" = torch.ops.aten.mm.default(view_231, permute_233);  view_231 = permute_233 = None
    view_232: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_65, [8, 768, 196]);  mm_65 = None
    permute_234: "f32[196, 196]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    permute_235: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_174: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    mul_367: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_174, primals_247);  primals_247 = None
    mul_368: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
    sum_63: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_197);  mul_367 = None
    sum_64: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_197, sum_64);  sum_64 = None
    sub_95: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_63);  mul_368 = sum_63 = None
    sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_370);  sub_95 = mul_370 = None
    mul_371: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_96);  div_12 = sub_96 = None
    mul_372: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_174, mul_197);  mul_197 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_174, [0, 1]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_5: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_365, mul_371], 2);  mul_365 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_374: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_375: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, view_146)
    mul_376: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_375, -0.5);  mul_375 = None
    exp_5: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_376);  mul_376 = None
    mul_377: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_378: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_146, mul_377);  view_146 = mul_377 = None
    add_228: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_374, mul_378);  mul_374 = mul_378 = None
    mul_379: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_5, add_228);  cat_5 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_233: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_379, [1568, 1536]);  mul_379 = None
    mm_66: "f32[1568, 256]" = torch.ops.aten.mm.default(view_233, permute_236);  permute_236 = None
    permute_237: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_67: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_237, view_145);  permute_237 = view_145 = None
    permute_238: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_67: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[1536]" = torch.ops.aten.view.default(sum_67, [1536]);  sum_67 = None
    permute_239: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_235: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_66, [8, 196, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_381: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_235, primals_243);  primals_243 = None
    mul_382: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_381, 256)
    sum_68: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_381, mul_192);  mul_381 = None
    sum_69: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_192, sum_69);  sum_69 = None
    sub_98: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_382, sum_68);  mul_382 = sum_68 = None
    sub_99: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_98, mul_384);  sub_98 = mul_384 = None
    mul_385: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_13, sub_99);  div_13 = sub_99 = None
    mul_386: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_235, mul_192);  mul_192 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_229: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_226, mul_385);  add_226 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_236: "f32[1568, 256]" = torch.ops.aten.view.default(add_229, [1568, 256])
    mm_68: "f32[1568, 768]" = torch.ops.aten.mm.default(view_236, permute_240);  permute_240 = None
    permute_241: "f32[256, 1568]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_69: "f32[256, 768]" = torch.ops.aten.mm.default(permute_241, view_143);  permute_241 = view_143 = None
    permute_242: "f32[768, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[256]" = torch.ops.aten.view.default(sum_72, [256]);  sum_72 = None
    permute_243: "f32[256, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_238: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_68, [8, 196, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_387: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_238, getitem_140);  getitem_140 = None
    mul_388: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_238, permute_119);  view_238 = permute_119 = None
    permute_244: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_387, [0, 2, 1]);  mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_73: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_244, [0, 1], True)
    view_239: "f32[196]" = torch.ops.aten.view.default(sum_73, [196]);  sum_73 = None
    clone_177: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_240: "f32[6144, 196]" = torch.ops.aten.view.default(clone_177, [6144, 196]);  clone_177 = None
    permute_245: "f32[196, 6144]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_70: "f32[196, 196]" = torch.ops.aten.mm.default(permute_245, view_141);  permute_245 = view_141 = None
    permute_246: "f32[196, 196]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    mm_71: "f32[6144, 196]" = torch.ops.aten.mm.default(view_240, permute_247);  view_240 = permute_247 = None
    view_241: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_71, [8, 768, 196]);  mm_71 = None
    permute_248: "f32[196, 196]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    permute_249: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_178: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    mul_390: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_178, primals_237);  primals_237 = None
    mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_390, 768)
    sum_74: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True)
    mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_390, mul_189);  mul_390 = None
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2], True);  mul_392 = None
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_189, sum_75);  sum_75 = None
    sub_101: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_391, sum_74);  mul_391 = sum_74 = None
    sub_102: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_393);  sub_101 = mul_393 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_102);  div_14 = sub_102 = None
    mul_395: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_178, mul_189);  mul_189 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 1]);  mul_395 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_178, [0, 1]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_6: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_388, mul_394], 2);  mul_388 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_397: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_163, 0.5);  add_163 = None
    mul_398: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, view_140)
    mul_399: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_398, -0.5);  mul_398 = None
    exp_6: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_399);  mul_399 = None
    mul_400: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_401: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_140, mul_400);  view_140 = mul_400 = None
    add_231: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_397, mul_401);  mul_397 = mul_401 = None
    mul_402: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_6, add_231);  cat_6 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_242: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_402, [1568, 1536]);  mul_402 = None
    mm_72: "f32[1568, 256]" = torch.ops.aten.mm.default(view_242, permute_250);  permute_250 = None
    permute_251: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_73: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_251, view_139);  permute_251 = view_139 = None
    permute_252: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_78: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[1536]" = torch.ops.aten.view.default(sum_78, [1536]);  sum_78 = None
    permute_253: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_244: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_72, [8, 196, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_404: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_244, primals_233);  primals_233 = None
    mul_405: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_404, 256)
    sum_79: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True)
    mul_406: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_404, mul_184);  mul_404 = None
    sum_80: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [2], True);  mul_406 = None
    mul_407: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_184, sum_80);  sum_80 = None
    sub_104: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_405, sum_79);  mul_405 = sum_79 = None
    sub_105: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_104, mul_407);  sub_104 = mul_407 = None
    mul_408: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_15, sub_105);  div_15 = sub_105 = None
    mul_409: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_244, mul_184);  mul_184 = None
    sum_81: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1]);  mul_409 = None
    sum_82: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_244, [0, 1]);  view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_232: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_229, mul_408);  add_229 = mul_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_245: "f32[1568, 256]" = torch.ops.aten.view.default(add_232, [1568, 256])
    mm_74: "f32[1568, 768]" = torch.ops.aten.mm.default(view_245, permute_254);  permute_254 = None
    permute_255: "f32[256, 1568]" = torch.ops.aten.permute.default(view_245, [1, 0])
    mm_75: "f32[256, 768]" = torch.ops.aten.mm.default(permute_255, view_137);  permute_255 = view_137 = None
    permute_256: "f32[768, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_83: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
    view_246: "f32[256]" = torch.ops.aten.view.default(sum_83, [256]);  sum_83 = None
    permute_257: "f32[256, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_247: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_74, [8, 196, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_410: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_247, getitem_134);  getitem_134 = None
    mul_411: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_247, permute_114);  view_247 = permute_114 = None
    permute_258: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_410, [0, 2, 1]);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_84: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_258, [0, 1], True)
    view_248: "f32[196]" = torch.ops.aten.view.default(sum_84, [196]);  sum_84 = None
    clone_181: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_249: "f32[6144, 196]" = torch.ops.aten.view.default(clone_181, [6144, 196]);  clone_181 = None
    permute_259: "f32[196, 6144]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_76: "f32[196, 196]" = torch.ops.aten.mm.default(permute_259, view_135);  permute_259 = view_135 = None
    permute_260: "f32[196, 196]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    mm_77: "f32[6144, 196]" = torch.ops.aten.mm.default(view_249, permute_261);  view_249 = permute_261 = None
    view_250: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_77, [8, 768, 196]);  mm_77 = None
    permute_262: "f32[196, 196]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    permute_263: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_182: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    mul_413: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_182, primals_227);  primals_227 = None
    mul_414: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_413, 768)
    sum_85: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True)
    mul_415: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_413, mul_181);  mul_413 = None
    sum_86: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True);  mul_415 = None
    mul_416: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, sum_86);  sum_86 = None
    sub_107: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_414, sum_85);  mul_414 = sum_85 = None
    sub_108: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_416);  sub_107 = mul_416 = None
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_108);  div_16 = sub_108 = None
    mul_418: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_182, mul_181);  mul_181 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 1]);  mul_418 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_182, [0, 1]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_7: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_411, mul_417], 2);  mul_411 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_420: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_156, 0.5);  add_156 = None
    mul_421: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, view_134)
    mul_422: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_7: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_424: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_134, mul_423);  view_134 = mul_423 = None
    add_234: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_7, add_234);  cat_7 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_251: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_425, [1568, 1536]);  mul_425 = None
    mm_78: "f32[1568, 256]" = torch.ops.aten.mm.default(view_251, permute_264);  permute_264 = None
    permute_265: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_251, [1, 0])
    mm_79: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_265, view_133);  permute_265 = view_133 = None
    permute_266: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_89: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
    view_252: "f32[1536]" = torch.ops.aten.view.default(sum_89, [1536]);  sum_89 = None
    permute_267: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_253: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_78, [8, 196, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_427: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_253, primals_223);  primals_223 = None
    mul_428: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_427, 256)
    sum_90: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_427, mul_176);  mul_427 = None
    sum_91: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_176, sum_91);  sum_91 = None
    sub_110: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_428, sum_90);  mul_428 = sum_90 = None
    sub_111: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_110, mul_430);  sub_110 = mul_430 = None
    mul_431: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_17, sub_111);  div_17 = sub_111 = None
    mul_432: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_253, mul_176);  mul_176 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_93: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_253, [0, 1]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_235: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_232, mul_431);  add_232 = mul_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_254: "f32[1568, 256]" = torch.ops.aten.view.default(add_235, [1568, 256])
    mm_80: "f32[1568, 768]" = torch.ops.aten.mm.default(view_254, permute_268);  permute_268 = None
    permute_269: "f32[256, 1568]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_81: "f32[256, 768]" = torch.ops.aten.mm.default(permute_269, view_131);  permute_269 = view_131 = None
    permute_270: "f32[768, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_94: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[256]" = torch.ops.aten.view.default(sum_94, [256]);  sum_94 = None
    permute_271: "f32[256, 768]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_256: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_80, [8, 196, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_433: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, getitem_128);  getitem_128 = None
    mul_434: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_256, permute_109);  view_256 = permute_109 = None
    permute_272: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_433, [0, 2, 1]);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_95: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_272, [0, 1], True)
    view_257: "f32[196]" = torch.ops.aten.view.default(sum_95, [196]);  sum_95 = None
    clone_185: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_258: "f32[6144, 196]" = torch.ops.aten.view.default(clone_185, [6144, 196]);  clone_185 = None
    permute_273: "f32[196, 6144]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_82: "f32[196, 196]" = torch.ops.aten.mm.default(permute_273, view_129);  permute_273 = view_129 = None
    permute_274: "f32[196, 196]" = torch.ops.aten.permute.default(mm_82, [1, 0]);  mm_82 = None
    mm_83: "f32[6144, 196]" = torch.ops.aten.mm.default(view_258, permute_275);  view_258 = permute_275 = None
    view_259: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_83, [8, 768, 196]);  mm_83 = None
    permute_276: "f32[196, 196]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    permute_277: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_186: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
    mul_436: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_186, primals_217);  primals_217 = None
    mul_437: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_436, 768)
    sum_96: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [2], True)
    mul_438: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_436, mul_173);  mul_436 = None
    sum_97: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [2], True);  mul_438 = None
    mul_439: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_173, sum_97);  sum_97 = None
    sub_113: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_437, sum_96);  mul_437 = sum_96 = None
    sub_114: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_439);  sub_113 = mul_439 = None
    mul_440: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_114);  div_18 = sub_114 = None
    mul_441: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_186, mul_173);  mul_173 = None
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 1]);  mul_441 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_186, [0, 1]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_8: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_434, mul_440], 2);  mul_434 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_443: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_149, 0.5);  add_149 = None
    mul_444: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_445: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_444, -0.5);  mul_444 = None
    exp_8: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_445);  mul_445 = None
    mul_446: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_447: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_128, mul_446);  view_128 = mul_446 = None
    add_237: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_443, mul_447);  mul_443 = mul_447 = None
    mul_448: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_8, add_237);  cat_8 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_260: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_448, [1568, 1536]);  mul_448 = None
    mm_84: "f32[1568, 256]" = torch.ops.aten.mm.default(view_260, permute_278);  permute_278 = None
    permute_279: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_85: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_279, view_127);  permute_279 = view_127 = None
    permute_280: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_100: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[1536]" = torch.ops.aten.view.default(sum_100, [1536]);  sum_100 = None
    permute_281: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_262: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_84, [8, 196, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_450: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_262, primals_213);  primals_213 = None
    mul_451: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_450, 256)
    sum_101: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_450, [2], True)
    mul_452: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_450, mul_168);  mul_450 = None
    sum_102: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [2], True);  mul_452 = None
    mul_453: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_168, sum_102);  sum_102 = None
    sub_116: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_451, sum_101);  mul_451 = sum_101 = None
    sub_117: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_116, mul_453);  sub_116 = mul_453 = None
    mul_454: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_19, sub_117);  div_19 = sub_117 = None
    mul_455: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_262, mul_168);  mul_168 = None
    sum_103: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 1]);  mul_455 = None
    sum_104: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_262, [0, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_238: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_235, mul_454);  add_235 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_263: "f32[1568, 256]" = torch.ops.aten.view.default(add_238, [1568, 256])
    mm_86: "f32[1568, 768]" = torch.ops.aten.mm.default(view_263, permute_282);  permute_282 = None
    permute_283: "f32[256, 1568]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_87: "f32[256, 768]" = torch.ops.aten.mm.default(permute_283, view_125);  permute_283 = view_125 = None
    permute_284: "f32[768, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_105: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[256]" = torch.ops.aten.view.default(sum_105, [256]);  sum_105 = None
    permute_285: "f32[256, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_265: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_86, [8, 196, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_456: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_265, getitem_122);  getitem_122 = None
    mul_457: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_265, permute_104);  view_265 = permute_104 = None
    permute_286: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_456, [0, 2, 1]);  mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_106: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_286, [0, 1], True)
    view_266: "f32[196]" = torch.ops.aten.view.default(sum_106, [196]);  sum_106 = None
    clone_189: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_267: "f32[6144, 196]" = torch.ops.aten.view.default(clone_189, [6144, 196]);  clone_189 = None
    permute_287: "f32[196, 6144]" = torch.ops.aten.permute.default(view_267, [1, 0])
    mm_88: "f32[196, 196]" = torch.ops.aten.mm.default(permute_287, view_123);  permute_287 = view_123 = None
    permute_288: "f32[196, 196]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    mm_89: "f32[6144, 196]" = torch.ops.aten.mm.default(view_267, permute_289);  view_267 = permute_289 = None
    view_268: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_89, [8, 768, 196]);  mm_89 = None
    permute_290: "f32[196, 196]" = torch.ops.aten.permute.default(permute_288, [1, 0]);  permute_288 = None
    permute_291: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_268, [0, 2, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_190: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_291, memory_format = torch.contiguous_format);  permute_291 = None
    mul_459: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_190, primals_207);  primals_207 = None
    mul_460: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, 768)
    sum_107: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, mul_165);  mul_459 = None
    sum_108: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, sum_108);  sum_108 = None
    sub_119: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_460, sum_107);  mul_460 = sum_107 = None
    sub_120: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_462);  sub_119 = mul_462 = None
    mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_120);  div_20 = sub_120 = None
    mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_190, mul_165);  mul_165 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_190, [0, 1]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_9: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_457, mul_463], 2);  mul_457 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_466: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_467: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, view_122)
    mul_468: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_467, -0.5);  mul_467 = None
    exp_9: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_468);  mul_468 = None
    mul_469: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_470: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_122, mul_469);  view_122 = mul_469 = None
    add_240: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_466, mul_470);  mul_466 = mul_470 = None
    mul_471: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_9, add_240);  cat_9 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_269: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_471, [1568, 1536]);  mul_471 = None
    mm_90: "f32[1568, 256]" = torch.ops.aten.mm.default(view_269, permute_292);  permute_292 = None
    permute_293: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_91: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_293, view_121);  permute_293 = view_121 = None
    permute_294: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_111: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[1536]" = torch.ops.aten.view.default(sum_111, [1536]);  sum_111 = None
    permute_295: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    view_271: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_90, [8, 196, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_473: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_271, primals_203);  primals_203 = None
    mul_474: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_473, 256)
    sum_112: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [2], True)
    mul_475: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_473, mul_160);  mul_473 = None
    sum_113: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True);  mul_475 = None
    mul_476: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_160, sum_113);  sum_113 = None
    sub_122: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_474, sum_112);  mul_474 = sum_112 = None
    sub_123: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_122, mul_476);  sub_122 = mul_476 = None
    mul_477: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_21, sub_123);  div_21 = sub_123 = None
    mul_478: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_271, mul_160);  mul_160 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 1]);  mul_478 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_241: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_238, mul_477);  add_238 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_272: "f32[1568, 256]" = torch.ops.aten.view.default(add_241, [1568, 256])
    mm_92: "f32[1568, 768]" = torch.ops.aten.mm.default(view_272, permute_296);  permute_296 = None
    permute_297: "f32[256, 1568]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_93: "f32[256, 768]" = torch.ops.aten.mm.default(permute_297, view_119);  permute_297 = view_119 = None
    permute_298: "f32[768, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_116: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[256]" = torch.ops.aten.view.default(sum_116, [256]);  sum_116 = None
    permute_299: "f32[256, 768]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    view_274: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_92, [8, 196, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_479: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_274, getitem_116);  getitem_116 = None
    mul_480: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_274, permute_99);  view_274 = permute_99 = None
    permute_300: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_479, [0, 2, 1]);  mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_117: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_300, [0, 1], True)
    view_275: "f32[196]" = torch.ops.aten.view.default(sum_117, [196]);  sum_117 = None
    clone_193: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    view_276: "f32[6144, 196]" = torch.ops.aten.view.default(clone_193, [6144, 196]);  clone_193 = None
    permute_301: "f32[196, 6144]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_94: "f32[196, 196]" = torch.ops.aten.mm.default(permute_301, view_117);  permute_301 = view_117 = None
    permute_302: "f32[196, 196]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    mm_95: "f32[6144, 196]" = torch.ops.aten.mm.default(view_276, permute_303);  view_276 = permute_303 = None
    view_277: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_95, [8, 768, 196]);  mm_95 = None
    permute_304: "f32[196, 196]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    permute_305: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_277, [0, 2, 1]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_194: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_305, memory_format = torch.contiguous_format);  permute_305 = None
    mul_482: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_194, primals_197);  primals_197 = None
    mul_483: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, 768)
    sum_118: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, mul_157);  mul_482 = None
    sum_119: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_157, sum_119);  sum_119 = None
    sub_125: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_483, sum_118);  mul_483 = sum_118 = None
    sub_126: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_485);  sub_125 = mul_485 = None
    mul_486: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_126);  div_22 = sub_126 = None
    mul_487: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_194, mul_157);  mul_157 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_194, [0, 1]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_10: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_480, mul_486], 2);  mul_480 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_489: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_135, 0.5);  add_135 = None
    mul_490: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, view_116)
    mul_491: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
    exp_10: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
    mul_492: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_493: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_116, mul_492);  view_116 = mul_492 = None
    add_243: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
    mul_494: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_10, add_243);  cat_10 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_278: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_494, [1568, 1536]);  mul_494 = None
    mm_96: "f32[1568, 256]" = torch.ops.aten.mm.default(view_278, permute_306);  permute_306 = None
    permute_307: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_97: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_307, view_115);  permute_307 = view_115 = None
    permute_308: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[1536]" = torch.ops.aten.view.default(sum_122, [1536]);  sum_122 = None
    permute_309: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_280: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_96, [8, 196, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_496: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_280, primals_193);  primals_193 = None
    mul_497: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_496, 256)
    sum_123: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
    mul_498: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_496, mul_152);  mul_496 = None
    sum_124: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
    mul_499: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_152, sum_124);  sum_124 = None
    sub_128: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_497, sum_123);  mul_497 = sum_123 = None
    sub_129: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_128, mul_499);  sub_128 = mul_499 = None
    mul_500: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_23, sub_129);  div_23 = sub_129 = None
    mul_501: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_280, mul_152);  mul_152 = None
    sum_125: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
    sum_126: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_244: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_241, mul_500);  add_241 = mul_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_281: "f32[1568, 256]" = torch.ops.aten.view.default(add_244, [1568, 256])
    mm_98: "f32[1568, 768]" = torch.ops.aten.mm.default(view_281, permute_310);  permute_310 = None
    permute_311: "f32[256, 1568]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_99: "f32[256, 768]" = torch.ops.aten.mm.default(permute_311, view_113);  permute_311 = view_113 = None
    permute_312: "f32[768, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_127: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[256]" = torch.ops.aten.view.default(sum_127, [256]);  sum_127 = None
    permute_313: "f32[256, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_283: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_98, [8, 196, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_502: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_283, getitem_110);  getitem_110 = None
    mul_503: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_283, permute_94);  view_283 = permute_94 = None
    permute_314: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_502, [0, 2, 1]);  mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_128: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_314, [0, 1], True)
    view_284: "f32[196]" = torch.ops.aten.view.default(sum_128, [196]);  sum_128 = None
    clone_197: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    view_285: "f32[6144, 196]" = torch.ops.aten.view.default(clone_197, [6144, 196]);  clone_197 = None
    permute_315: "f32[196, 6144]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_100: "f32[196, 196]" = torch.ops.aten.mm.default(permute_315, view_111);  permute_315 = view_111 = None
    permute_316: "f32[196, 196]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    mm_101: "f32[6144, 196]" = torch.ops.aten.mm.default(view_285, permute_317);  view_285 = permute_317 = None
    view_286: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_101, [8, 768, 196]);  mm_101 = None
    permute_318: "f32[196, 196]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    permute_319: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_198: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    mul_505: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_198, primals_187);  primals_187 = None
    mul_506: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_505, 768)
    sum_129: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True)
    mul_507: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_505, mul_149);  mul_505 = None
    sum_130: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True);  mul_507 = None
    mul_508: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_149, sum_130);  sum_130 = None
    sub_131: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_506, sum_129);  mul_506 = sum_129 = None
    sub_132: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_131, mul_508);  sub_131 = mul_508 = None
    mul_509: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_132);  div_24 = sub_132 = None
    mul_510: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_198, mul_149);  mul_149 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1]);  mul_510 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_198, [0, 1]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_11: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_503, mul_509], 2);  mul_503 = mul_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_512: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_513: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, view_110)
    mul_514: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_513, -0.5);  mul_513 = None
    exp_11: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_514);  mul_514 = None
    mul_515: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_516: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_110, mul_515);  view_110 = mul_515 = None
    add_246: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_512, mul_516);  mul_512 = mul_516 = None
    mul_517: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_11, add_246);  cat_11 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_287: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_517, [1568, 1536]);  mul_517 = None
    mm_102: "f32[1568, 256]" = torch.ops.aten.mm.default(view_287, permute_320);  permute_320 = None
    permute_321: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_103: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_321, view_109);  permute_321 = view_109 = None
    permute_322: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_133: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[1536]" = torch.ops.aten.view.default(sum_133, [1536]);  sum_133 = None
    permute_323: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
    view_289: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_102, [8, 196, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_519: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_289, primals_183);  primals_183 = None
    mul_520: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_519, 256)
    sum_134: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True)
    mul_521: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_519, mul_144);  mul_519 = None
    sum_135: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [2], True);  mul_521 = None
    mul_522: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_144, sum_135);  sum_135 = None
    sub_134: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_520, sum_134);  mul_520 = sum_134 = None
    sub_135: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_134, mul_522);  sub_134 = mul_522 = None
    mul_523: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_135);  div_25 = sub_135 = None
    mul_524: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_289, mul_144);  mul_144 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_524, [0, 1]);  mul_524 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_247: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_244, mul_523);  add_244 = mul_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_290: "f32[1568, 256]" = torch.ops.aten.view.default(add_247, [1568, 256])
    mm_104: "f32[1568, 768]" = torch.ops.aten.mm.default(view_290, permute_324);  permute_324 = None
    permute_325: "f32[256, 1568]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_105: "f32[256, 768]" = torch.ops.aten.mm.default(permute_325, view_107);  permute_325 = view_107 = None
    permute_326: "f32[768, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_138: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[256]" = torch.ops.aten.view.default(sum_138, [256]);  sum_138 = None
    permute_327: "f32[256, 768]" = torch.ops.aten.permute.default(permute_326, [1, 0]);  permute_326 = None
    view_292: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_104, [8, 196, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_525: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, getitem_104);  getitem_104 = None
    mul_526: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_292, permute_89);  view_292 = permute_89 = None
    permute_328: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_525, [0, 2, 1]);  mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_139: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_328, [0, 1], True)
    view_293: "f32[196]" = torch.ops.aten.view.default(sum_139, [196]);  sum_139 = None
    clone_201: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
    view_294: "f32[6144, 196]" = torch.ops.aten.view.default(clone_201, [6144, 196]);  clone_201 = None
    permute_329: "f32[196, 6144]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_106: "f32[196, 196]" = torch.ops.aten.mm.default(permute_329, view_105);  permute_329 = view_105 = None
    permute_330: "f32[196, 196]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    mm_107: "f32[6144, 196]" = torch.ops.aten.mm.default(view_294, permute_331);  view_294 = permute_331 = None
    view_295: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_107, [8, 768, 196]);  mm_107 = None
    permute_332: "f32[196, 196]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    permute_333: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_295, [0, 2, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_202: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_333, memory_format = torch.contiguous_format);  permute_333 = None
    mul_528: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_202, primals_177);  primals_177 = None
    mul_529: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_528, 768)
    sum_140: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [2], True)
    mul_530: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_528, mul_141);  mul_528 = None
    sum_141: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
    mul_531: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_141, sum_141);  sum_141 = None
    sub_137: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_529, sum_140);  mul_529 = sum_140 = None
    sub_138: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_137, mul_531);  sub_137 = mul_531 = None
    mul_532: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_138);  div_26 = sub_138 = None
    mul_533: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_202, mul_141);  mul_141 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_533, [0, 1]);  mul_533 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_202, [0, 1]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_12: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_526, mul_532], 2);  mul_526 = mul_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_535: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
    mul_536: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, view_104)
    mul_537: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_536, -0.5);  mul_536 = None
    exp_12: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_537);  mul_537 = None
    mul_538: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_539: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_104, mul_538);  view_104 = mul_538 = None
    add_249: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_535, mul_539);  mul_535 = mul_539 = None
    mul_540: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_12, add_249);  cat_12 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_296: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_540, [1568, 1536]);  mul_540 = None
    mm_108: "f32[1568, 256]" = torch.ops.aten.mm.default(view_296, permute_334);  permute_334 = None
    permute_335: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_109: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_335, view_103);  permute_335 = view_103 = None
    permute_336: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_144: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[1536]" = torch.ops.aten.view.default(sum_144, [1536]);  sum_144 = None
    permute_337: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_298: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_108, [8, 196, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_542: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_298, primals_173);  primals_173 = None
    mul_543: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_542, 256)
    sum_145: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_542, [2], True)
    mul_544: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_542, mul_136);  mul_542 = None
    sum_146: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_544, [2], True);  mul_544 = None
    mul_545: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_136, sum_146);  sum_146 = None
    sub_140: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_543, sum_145);  mul_543 = sum_145 = None
    sub_141: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_140, mul_545);  sub_140 = mul_545 = None
    mul_546: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_141);  div_27 = sub_141 = None
    mul_547: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_298, mul_136);  mul_136 = None
    sum_147: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 1]);  mul_547 = None
    sum_148: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_298, [0, 1]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_250: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_247, mul_546);  add_247 = mul_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_299: "f32[1568, 256]" = torch.ops.aten.view.default(add_250, [1568, 256])
    mm_110: "f32[1568, 768]" = torch.ops.aten.mm.default(view_299, permute_338);  permute_338 = None
    permute_339: "f32[256, 1568]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_111: "f32[256, 768]" = torch.ops.aten.mm.default(permute_339, view_101);  permute_339 = view_101 = None
    permute_340: "f32[768, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_149: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[256]" = torch.ops.aten.view.default(sum_149, [256]);  sum_149 = None
    permute_341: "f32[256, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_301: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_110, [8, 196, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_548: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_301, getitem_98);  getitem_98 = None
    mul_549: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_301, permute_84);  view_301 = permute_84 = None
    permute_342: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_548, [0, 2, 1]);  mul_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_150: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_342, [0, 1], True)
    view_302: "f32[196]" = torch.ops.aten.view.default(sum_150, [196]);  sum_150 = None
    clone_205: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
    view_303: "f32[6144, 196]" = torch.ops.aten.view.default(clone_205, [6144, 196]);  clone_205 = None
    permute_343: "f32[196, 6144]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_112: "f32[196, 196]" = torch.ops.aten.mm.default(permute_343, view_99);  permute_343 = view_99 = None
    permute_344: "f32[196, 196]" = torch.ops.aten.permute.default(mm_112, [1, 0]);  mm_112 = None
    mm_113: "f32[6144, 196]" = torch.ops.aten.mm.default(view_303, permute_345);  view_303 = permute_345 = None
    view_304: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_113, [8, 768, 196]);  mm_113 = None
    permute_346: "f32[196, 196]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    permute_347: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_206: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
    mul_551: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_206, primals_167);  primals_167 = None
    mul_552: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_551, 768)
    sum_151: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
    mul_553: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_551, mul_133);  mul_551 = None
    sum_152: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
    mul_554: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_133, sum_152);  sum_152 = None
    sub_143: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_552, sum_151);  mul_552 = sum_151 = None
    sub_144: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_143, mul_554);  sub_143 = mul_554 = None
    mul_555: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_144);  div_28 = sub_144 = None
    mul_556: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_206, mul_133);  mul_133 = None
    sum_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_206, [0, 1]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_13: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_549, mul_555], 2);  mul_549 = mul_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_558: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_559: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_560: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_559, -0.5);  mul_559 = None
    exp_13: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_560);  mul_560 = None
    mul_561: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_562: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_98, mul_561);  view_98 = mul_561 = None
    add_252: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_558, mul_562);  mul_558 = mul_562 = None
    mul_563: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_13, add_252);  cat_13 = add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_305: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_563, [1568, 1536]);  mul_563 = None
    mm_114: "f32[1568, 256]" = torch.ops.aten.mm.default(view_305, permute_348);  permute_348 = None
    permute_349: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_115: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_349, view_97);  permute_349 = view_97 = None
    permute_350: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[1536]" = torch.ops.aten.view.default(sum_155, [1536]);  sum_155 = None
    permute_351: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_307: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_114, [8, 196, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_565: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_307, primals_163);  primals_163 = None
    mul_566: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_565, 256)
    sum_156: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True)
    mul_567: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_565, mul_128);  mul_565 = None
    sum_157: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_567, [2], True);  mul_567 = None
    mul_568: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_128, sum_157);  sum_157 = None
    sub_146: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_566, sum_156);  mul_566 = sum_156 = None
    sub_147: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_146, mul_568);  sub_146 = mul_568 = None
    mul_569: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_29, sub_147);  div_29 = sub_147 = None
    mul_570: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_307, mul_128);  mul_128 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 1]);  mul_570 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_253: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_250, mul_569);  add_250 = mul_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_308: "f32[1568, 256]" = torch.ops.aten.view.default(add_253, [1568, 256])
    mm_116: "f32[1568, 768]" = torch.ops.aten.mm.default(view_308, permute_352);  permute_352 = None
    permute_353: "f32[256, 1568]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_117: "f32[256, 768]" = torch.ops.aten.mm.default(permute_353, view_95);  permute_353 = view_95 = None
    permute_354: "f32[768, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_160: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[256]" = torch.ops.aten.view.default(sum_160, [256]);  sum_160 = None
    permute_355: "f32[256, 768]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
    view_310: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_116, [8, 196, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_571: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_310, getitem_92);  getitem_92 = None
    mul_572: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_310, permute_79);  view_310 = permute_79 = None
    permute_356: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_571, [0, 2, 1]);  mul_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_161: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_356, [0, 1], True)
    view_311: "f32[196]" = torch.ops.aten.view.default(sum_161, [196]);  sum_161 = None
    clone_209: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_356, memory_format = torch.contiguous_format);  permute_356 = None
    view_312: "f32[6144, 196]" = torch.ops.aten.view.default(clone_209, [6144, 196]);  clone_209 = None
    permute_357: "f32[196, 6144]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_118: "f32[196, 196]" = torch.ops.aten.mm.default(permute_357, view_93);  permute_357 = view_93 = None
    permute_358: "f32[196, 196]" = torch.ops.aten.permute.default(mm_118, [1, 0]);  mm_118 = None
    mm_119: "f32[6144, 196]" = torch.ops.aten.mm.default(view_312, permute_359);  view_312 = permute_359 = None
    view_313: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_119, [8, 768, 196]);  mm_119 = None
    permute_360: "f32[196, 196]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    permute_361: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_210: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    mul_574: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_210, primals_157);  primals_157 = None
    mul_575: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_574, 768)
    sum_162: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True)
    mul_576: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_574, mul_125);  mul_574 = None
    sum_163: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True);  mul_576 = None
    mul_577: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_125, sum_163);  sum_163 = None
    sub_149: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_575, sum_162);  mul_575 = sum_162 = None
    sub_150: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_149, mul_577);  sub_149 = mul_577 = None
    mul_578: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_150);  div_30 = sub_150 = None
    mul_579: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_210, mul_125);  mul_125 = None
    sum_164: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1]);  mul_579 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_210, [0, 1]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_14: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_572, mul_578], 2);  mul_572 = mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_581: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_107, 0.5);  add_107 = None
    mul_582: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, view_92)
    mul_583: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_582, -0.5);  mul_582 = None
    exp_14: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_583);  mul_583 = None
    mul_584: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_585: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_92, mul_584);  view_92 = mul_584 = None
    add_255: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_581, mul_585);  mul_581 = mul_585 = None
    mul_586: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_14, add_255);  cat_14 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_314: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_586, [1568, 1536]);  mul_586 = None
    mm_120: "f32[1568, 256]" = torch.ops.aten.mm.default(view_314, permute_362);  permute_362 = None
    permute_363: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_121: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_363, view_91);  permute_363 = view_91 = None
    permute_364: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_166: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[1536]" = torch.ops.aten.view.default(sum_166, [1536]);  sum_166 = None
    permute_365: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_316: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_120, [8, 196, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_588: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_316, primals_153);  primals_153 = None
    mul_589: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_588, 256)
    sum_167: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_588, [2], True)
    mul_590: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_588, mul_120);  mul_588 = None
    sum_168: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True);  mul_590 = None
    mul_591: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_120, sum_168);  sum_168 = None
    sub_152: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_589, sum_167);  mul_589 = sum_167 = None
    sub_153: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_152, mul_591);  sub_152 = mul_591 = None
    mul_592: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_153);  div_31 = sub_153 = None
    mul_593: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_316, mul_120);  mul_120 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 1]);  mul_593 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_316, [0, 1]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_256: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_253, mul_592);  add_253 = mul_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_317: "f32[1568, 256]" = torch.ops.aten.view.default(add_256, [1568, 256])
    mm_122: "f32[1568, 768]" = torch.ops.aten.mm.default(view_317, permute_366);  permute_366 = None
    permute_367: "f32[256, 1568]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_123: "f32[256, 768]" = torch.ops.aten.mm.default(permute_367, view_89);  permute_367 = view_89 = None
    permute_368: "f32[768, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_369: "f32[256, 768]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_319: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_122, [8, 196, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_594: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_319, getitem_86);  getitem_86 = None
    mul_595: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_319, permute_74);  view_319 = permute_74 = None
    permute_370: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_594, [0, 2, 1]);  mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_172: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_370, [0, 1], True)
    view_320: "f32[196]" = torch.ops.aten.view.default(sum_172, [196]);  sum_172 = None
    clone_213: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
    view_321: "f32[6144, 196]" = torch.ops.aten.view.default(clone_213, [6144, 196]);  clone_213 = None
    permute_371: "f32[196, 6144]" = torch.ops.aten.permute.default(view_321, [1, 0])
    mm_124: "f32[196, 196]" = torch.ops.aten.mm.default(permute_371, view_87);  permute_371 = view_87 = None
    permute_372: "f32[196, 196]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    mm_125: "f32[6144, 196]" = torch.ops.aten.mm.default(view_321, permute_373);  view_321 = permute_373 = None
    view_322: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_125, [8, 768, 196]);  mm_125 = None
    permute_374: "f32[196, 196]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    permute_375: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_214: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    mul_597: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_214, primals_147);  primals_147 = None
    mul_598: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_597, 768)
    sum_173: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True)
    mul_599: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_597, mul_117);  mul_597 = None
    sum_174: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True);  mul_599 = None
    mul_600: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_117, sum_174);  sum_174 = None
    sub_155: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_598, sum_173);  mul_598 = sum_173 = None
    sub_156: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_600);  sub_155 = mul_600 = None
    mul_601: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_156);  div_32 = sub_156 = None
    mul_602: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_214, mul_117);  mul_117 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 1]);  mul_602 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_214, [0, 1]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_15: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_595, mul_601], 2);  mul_595 = mul_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_604: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_100, 0.5);  add_100 = None
    mul_605: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, view_86)
    mul_606: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_605, -0.5);  mul_605 = None
    exp_15: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_606);  mul_606 = None
    mul_607: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_608: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_86, mul_607);  view_86 = mul_607 = None
    add_258: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_604, mul_608);  mul_604 = mul_608 = None
    mul_609: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_15, add_258);  cat_15 = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_323: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_609, [1568, 1536]);  mul_609 = None
    mm_126: "f32[1568, 256]" = torch.ops.aten.mm.default(view_323, permute_376);  permute_376 = None
    permute_377: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_127: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_377, view_85);  permute_377 = view_85 = None
    permute_378: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_177: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[1536]" = torch.ops.aten.view.default(sum_177, [1536]);  sum_177 = None
    permute_379: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_325: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_126, [8, 196, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_611: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_325, primals_143);  primals_143 = None
    mul_612: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_611, 256)
    sum_178: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True)
    mul_613: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_611, mul_112);  mul_611 = None
    sum_179: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True);  mul_613 = None
    mul_614: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_112, sum_179);  sum_179 = None
    sub_158: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_612, sum_178);  mul_612 = sum_178 = None
    sub_159: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_158, mul_614);  sub_158 = mul_614 = None
    mul_615: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_159);  div_33 = sub_159 = None
    mul_616: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_325, mul_112);  mul_112 = None
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 1]);  mul_616 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_325, [0, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_259: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_256, mul_615);  add_256 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_326: "f32[1568, 256]" = torch.ops.aten.view.default(add_259, [1568, 256])
    mm_128: "f32[1568, 768]" = torch.ops.aten.mm.default(view_326, permute_380);  permute_380 = None
    permute_381: "f32[256, 1568]" = torch.ops.aten.permute.default(view_326, [1, 0])
    mm_129: "f32[256, 768]" = torch.ops.aten.mm.default(permute_381, view_83);  permute_381 = view_83 = None
    permute_382: "f32[768, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_182: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_326, [0], True);  view_326 = None
    view_327: "f32[256]" = torch.ops.aten.view.default(sum_182, [256]);  sum_182 = None
    permute_383: "f32[256, 768]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_328: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_128, [8, 196, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_617: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, getitem_80);  getitem_80 = None
    mul_618: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_328, permute_69);  view_328 = permute_69 = None
    permute_384: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_617, [0, 2, 1]);  mul_617 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_183: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_384, [0, 1], True)
    view_329: "f32[196]" = torch.ops.aten.view.default(sum_183, [196]);  sum_183 = None
    clone_217: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_384, memory_format = torch.contiguous_format);  permute_384 = None
    view_330: "f32[6144, 196]" = torch.ops.aten.view.default(clone_217, [6144, 196]);  clone_217 = None
    permute_385: "f32[196, 6144]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_130: "f32[196, 196]" = torch.ops.aten.mm.default(permute_385, view_81);  permute_385 = view_81 = None
    permute_386: "f32[196, 196]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[6144, 196]" = torch.ops.aten.mm.default(view_330, permute_387);  view_330 = permute_387 = None
    view_331: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_131, [8, 768, 196]);  mm_131 = None
    permute_388: "f32[196, 196]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    permute_389: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_218: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
    mul_620: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_218, primals_137);  primals_137 = None
    mul_621: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_620, 768)
    sum_184: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True)
    mul_622: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_620, mul_109);  mul_620 = None
    sum_185: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True);  mul_622 = None
    mul_623: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_109, sum_185);  sum_185 = None
    sub_161: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_621, sum_184);  mul_621 = sum_184 = None
    sub_162: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_161, mul_623);  sub_161 = mul_623 = None
    mul_624: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_162);  div_34 = sub_162 = None
    mul_625: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_218, mul_109);  mul_109 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 1]);  mul_625 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_218, [0, 1]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_16: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_618, mul_624], 2);  mul_618 = mul_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_627: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_93, 0.5);  add_93 = None
    mul_628: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, view_80)
    mul_629: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_628, -0.5);  mul_628 = None
    exp_16: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_629);  mul_629 = None
    mul_630: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_631: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_80, mul_630);  view_80 = mul_630 = None
    add_261: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_627, mul_631);  mul_627 = mul_631 = None
    mul_632: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_16, add_261);  cat_16 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_332: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_632, [1568, 1536]);  mul_632 = None
    mm_132: "f32[1568, 256]" = torch.ops.aten.mm.default(view_332, permute_390);  permute_390 = None
    permute_391: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_133: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_391, view_79);  permute_391 = view_79 = None
    permute_392: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_188: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[1536]" = torch.ops.aten.view.default(sum_188, [1536]);  sum_188 = None
    permute_393: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_334: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_132, [8, 196, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_634: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_334, primals_133);  primals_133 = None
    mul_635: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_634, 256)
    sum_189: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [2], True)
    mul_636: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_634, mul_104);  mul_634 = None
    sum_190: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_636, [2], True);  mul_636 = None
    mul_637: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_104, sum_190);  sum_190 = None
    sub_164: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_635, sum_189);  mul_635 = sum_189 = None
    sub_165: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_164, mul_637);  sub_164 = mul_637 = None
    mul_638: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_35, sub_165);  div_35 = sub_165 = None
    mul_639: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_334, mul_104);  mul_104 = None
    sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 1]);  mul_639 = None
    sum_192: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_334, [0, 1]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_262: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_259, mul_638);  add_259 = mul_638 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_335: "f32[1568, 256]" = torch.ops.aten.view.default(add_262, [1568, 256])
    mm_134: "f32[1568, 768]" = torch.ops.aten.mm.default(view_335, permute_394);  permute_394 = None
    permute_395: "f32[256, 1568]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_135: "f32[256, 768]" = torch.ops.aten.mm.default(permute_395, view_77);  permute_395 = view_77 = None
    permute_396: "f32[768, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_193: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[256]" = torch.ops.aten.view.default(sum_193, [256]);  sum_193 = None
    permute_397: "f32[256, 768]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_337: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_134, [8, 196, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_640: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_337, getitem_74);  getitem_74 = None
    mul_641: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_337, permute_64);  view_337 = permute_64 = None
    permute_398: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_640, [0, 2, 1]);  mul_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_194: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_398, [0, 1], True)
    view_338: "f32[196]" = torch.ops.aten.view.default(sum_194, [196]);  sum_194 = None
    clone_221: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None
    view_339: "f32[6144, 196]" = torch.ops.aten.view.default(clone_221, [6144, 196]);  clone_221 = None
    permute_399: "f32[196, 6144]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_136: "f32[196, 196]" = torch.ops.aten.mm.default(permute_399, view_75);  permute_399 = view_75 = None
    permute_400: "f32[196, 196]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[6144, 196]" = torch.ops.aten.mm.default(view_339, permute_401);  view_339 = permute_401 = None
    view_340: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_137, [8, 768, 196]);  mm_137 = None
    permute_402: "f32[196, 196]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    permute_403: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_340, [0, 2, 1]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_222: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
    mul_643: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_222, primals_127);  primals_127 = None
    mul_644: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_643, 768)
    sum_195: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_643, [2], True)
    mul_645: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_643, mul_101);  mul_643 = None
    sum_196: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_645, [2], True);  mul_645 = None
    mul_646: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_101, sum_196);  sum_196 = None
    sub_167: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_644, sum_195);  mul_644 = sum_195 = None
    sub_168: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_167, mul_646);  sub_167 = mul_646 = None
    mul_647: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_168);  div_36 = sub_168 = None
    mul_648: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_222, mul_101);  mul_101 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_648, [0, 1]);  mul_648 = None
    sum_198: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_222, [0, 1]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_17: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_641, mul_647], 2);  mul_641 = mul_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_650: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_651: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, view_74)
    mul_652: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_651, -0.5);  mul_651 = None
    exp_17: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_652);  mul_652 = None
    mul_653: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_654: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_74, mul_653);  view_74 = mul_653 = None
    add_264: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_650, mul_654);  mul_650 = mul_654 = None
    mul_655: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_17, add_264);  cat_17 = add_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_341: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_655, [1568, 1536]);  mul_655 = None
    mm_138: "f32[1568, 256]" = torch.ops.aten.mm.default(view_341, permute_404);  permute_404 = None
    permute_405: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_139: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_405, view_73);  permute_405 = view_73 = None
    permute_406: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_199: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[1536]" = torch.ops.aten.view.default(sum_199, [1536]);  sum_199 = None
    permute_407: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_343: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_138, [8, 196, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_657: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_343, primals_123);  primals_123 = None
    mul_658: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_657, 256)
    sum_200: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_657, [2], True)
    mul_659: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_657, mul_96);  mul_657 = None
    sum_201: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_659, [2], True);  mul_659 = None
    mul_660: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_96, sum_201);  sum_201 = None
    sub_170: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_658, sum_200);  mul_658 = sum_200 = None
    sub_171: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_170, mul_660);  sub_170 = mul_660 = None
    mul_661: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_171);  div_37 = sub_171 = None
    mul_662: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_343, mul_96);  mul_96 = None
    sum_202: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 1]);  mul_662 = None
    sum_203: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_265: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_262, mul_661);  add_262 = mul_661 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_344: "f32[1568, 256]" = torch.ops.aten.view.default(add_265, [1568, 256])
    mm_140: "f32[1568, 768]" = torch.ops.aten.mm.default(view_344, permute_408);  permute_408 = None
    permute_409: "f32[256, 1568]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_141: "f32[256, 768]" = torch.ops.aten.mm.default(permute_409, view_71);  permute_409 = view_71 = None
    permute_410: "f32[768, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_204: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[256]" = torch.ops.aten.view.default(sum_204, [256]);  sum_204 = None
    permute_411: "f32[256, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_346: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_140, [8, 196, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_663: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_346, getitem_68);  getitem_68 = None
    mul_664: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_346, permute_59);  view_346 = permute_59 = None
    permute_412: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_663, [0, 2, 1]);  mul_663 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_205: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_412, [0, 1], True)
    view_347: "f32[196]" = torch.ops.aten.view.default(sum_205, [196]);  sum_205 = None
    clone_225: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_348: "f32[6144, 196]" = torch.ops.aten.view.default(clone_225, [6144, 196]);  clone_225 = None
    permute_413: "f32[196, 6144]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_142: "f32[196, 196]" = torch.ops.aten.mm.default(permute_413, view_69);  permute_413 = view_69 = None
    permute_414: "f32[196, 196]" = torch.ops.aten.permute.default(mm_142, [1, 0]);  mm_142 = None
    mm_143: "f32[6144, 196]" = torch.ops.aten.mm.default(view_348, permute_415);  view_348 = permute_415 = None
    view_349: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_143, [8, 768, 196]);  mm_143 = None
    permute_416: "f32[196, 196]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    permute_417: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_226: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
    mul_666: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_226, primals_117);  primals_117 = None
    mul_667: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_666, 768)
    sum_206: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True)
    mul_668: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_666, mul_93);  mul_666 = None
    sum_207: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [2], True);  mul_668 = None
    mul_669: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_93, sum_207);  sum_207 = None
    sub_173: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_667, sum_206);  mul_667 = sum_206 = None
    sub_174: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_173, mul_669);  sub_173 = mul_669 = None
    mul_670: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_174);  div_38 = sub_174 = None
    mul_671: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_226, mul_93);  mul_93 = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1]);  mul_671 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_226, [0, 1]);  clone_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_18: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_664, mul_670], 2);  mul_664 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_673: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_79, 0.5);  add_79 = None
    mul_674: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_675: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_674, -0.5);  mul_674 = None
    exp_18: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_675);  mul_675 = None
    mul_676: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_677: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_68, mul_676);  view_68 = mul_676 = None
    add_267: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_673, mul_677);  mul_673 = mul_677 = None
    mul_678: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_18, add_267);  cat_18 = add_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_350: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_678, [1568, 1536]);  mul_678 = None
    mm_144: "f32[1568, 256]" = torch.ops.aten.mm.default(view_350, permute_418);  permute_418 = None
    permute_419: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_145: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_419, view_67);  permute_419 = view_67 = None
    permute_420: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_210: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[1536]" = torch.ops.aten.view.default(sum_210, [1536]);  sum_210 = None
    permute_421: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
    view_352: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_144, [8, 196, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_680: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_352, primals_113);  primals_113 = None
    mul_681: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_680, 256)
    sum_211: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_680, [2], True)
    mul_682: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_680, mul_88);  mul_680 = None
    sum_212: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [2], True);  mul_682 = None
    mul_683: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_88, sum_212);  sum_212 = None
    sub_176: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_681, sum_211);  mul_681 = sum_211 = None
    sub_177: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_176, mul_683);  sub_176 = mul_683 = None
    mul_684: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_177);  div_39 = sub_177 = None
    mul_685: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_352, mul_88);  mul_88 = None
    sum_213: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 1]);  mul_685 = None
    sum_214: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_352, [0, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_268: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_265, mul_684);  add_265 = mul_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_353: "f32[1568, 256]" = torch.ops.aten.view.default(add_268, [1568, 256])
    mm_146: "f32[1568, 768]" = torch.ops.aten.mm.default(view_353, permute_422);  permute_422 = None
    permute_423: "f32[256, 1568]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_147: "f32[256, 768]" = torch.ops.aten.mm.default(permute_423, view_65);  permute_423 = view_65 = None
    permute_424: "f32[768, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_215: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.view.default(sum_215, [256]);  sum_215 = None
    permute_425: "f32[256, 768]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_355: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_146, [8, 196, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_686: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_355, getitem_62);  getitem_62 = None
    mul_687: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_355, permute_54);  view_355 = permute_54 = None
    permute_426: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_686, [0, 2, 1]);  mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_216: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_426, [0, 1], True)
    view_356: "f32[196]" = torch.ops.aten.view.default(sum_216, [196]);  sum_216 = None
    clone_229: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_357: "f32[6144, 196]" = torch.ops.aten.view.default(clone_229, [6144, 196]);  clone_229 = None
    permute_427: "f32[196, 6144]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_148: "f32[196, 196]" = torch.ops.aten.mm.default(permute_427, view_63);  permute_427 = view_63 = None
    permute_428: "f32[196, 196]" = torch.ops.aten.permute.default(mm_148, [1, 0]);  mm_148 = None
    mm_149: "f32[6144, 196]" = torch.ops.aten.mm.default(view_357, permute_429);  view_357 = permute_429 = None
    view_358: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_149, [8, 768, 196]);  mm_149 = None
    permute_430: "f32[196, 196]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    permute_431: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_230: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_431, memory_format = torch.contiguous_format);  permute_431 = None
    mul_689: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_230, primals_107);  primals_107 = None
    mul_690: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_689, 768)
    sum_217: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True)
    mul_691: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_689, mul_85);  mul_689 = None
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True);  mul_691 = None
    mul_692: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_218);  sum_218 = None
    sub_179: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_690, sum_217);  mul_690 = sum_217 = None
    sub_180: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_179, mul_692);  sub_179 = mul_692 = None
    mul_693: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_180);  div_40 = sub_180 = None
    mul_694: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_230, mul_85);  mul_85 = None
    sum_219: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 1]);  mul_694 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_230, [0, 1]);  clone_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_19: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_687, mul_693], 2);  mul_687 = mul_693 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_696: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_697: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, view_62)
    mul_698: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_697, -0.5);  mul_697 = None
    exp_19: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_698);  mul_698 = None
    mul_699: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_700: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_62, mul_699);  view_62 = mul_699 = None
    add_270: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_696, mul_700);  mul_696 = mul_700 = None
    mul_701: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_19, add_270);  cat_19 = add_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_359: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_701, [1568, 1536]);  mul_701 = None
    mm_150: "f32[1568, 256]" = torch.ops.aten.mm.default(view_359, permute_432);  permute_432 = None
    permute_433: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_151: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_433, view_61);  permute_433 = view_61 = None
    permute_434: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_221: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[1536]" = torch.ops.aten.view.default(sum_221, [1536]);  sum_221 = None
    permute_435: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_361: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_150, [8, 196, 256]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_703: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_361, primals_103);  primals_103 = None
    mul_704: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_703, 256)
    sum_222: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
    mul_705: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_703, mul_80);  mul_703 = None
    sum_223: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
    mul_706: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_80, sum_223);  sum_223 = None
    sub_182: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_704, sum_222);  mul_704 = sum_222 = None
    sub_183: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_182, mul_706);  sub_182 = mul_706 = None
    mul_707: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_41, sub_183);  div_41 = sub_183 = None
    mul_708: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_361, mul_80);  mul_80 = None
    sum_224: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
    sum_225: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_271: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_268, mul_707);  add_268 = mul_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_362: "f32[1568, 256]" = torch.ops.aten.view.default(add_271, [1568, 256])
    mm_152: "f32[1568, 768]" = torch.ops.aten.mm.default(view_362, permute_436);  permute_436 = None
    permute_437: "f32[256, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_153: "f32[256, 768]" = torch.ops.aten.mm.default(permute_437, view_59);  permute_437 = view_59 = None
    permute_438: "f32[768, 256]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_226: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[256]" = torch.ops.aten.view.default(sum_226, [256]);  sum_226 = None
    permute_439: "f32[256, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_364: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_152, [8, 196, 768]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_709: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, getitem_56);  getitem_56 = None
    mul_710: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, permute_49);  view_364 = permute_49 = None
    permute_440: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_709, [0, 2, 1]);  mul_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_227: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_440, [0, 1], True)
    view_365: "f32[196]" = torch.ops.aten.view.default(sum_227, [196]);  sum_227 = None
    clone_233: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
    view_366: "f32[6144, 196]" = torch.ops.aten.view.default(clone_233, [6144, 196]);  clone_233 = None
    permute_441: "f32[196, 6144]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_154: "f32[196, 196]" = torch.ops.aten.mm.default(permute_441, view_57);  permute_441 = view_57 = None
    permute_442: "f32[196, 196]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    mm_155: "f32[6144, 196]" = torch.ops.aten.mm.default(view_366, permute_443);  view_366 = permute_443 = None
    view_367: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_155, [8, 768, 196]);  mm_155 = None
    permute_444: "f32[196, 196]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    permute_445: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_367, [0, 2, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_234: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_445, memory_format = torch.contiguous_format);  permute_445 = None
    mul_712: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_234, primals_97);  primals_97 = None
    mul_713: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_712, 768)
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_712, [2], True)
    mul_714: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_712, mul_77);  mul_712 = None
    sum_229: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_714, [2], True);  mul_714 = None
    mul_715: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_229);  sum_229 = None
    sub_185: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_713, sum_228);  mul_713 = sum_228 = None
    sub_186: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_185, mul_715);  sub_185 = mul_715 = None
    mul_716: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_186);  div_42 = sub_186 = None
    mul_717: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_234, mul_77);  mul_77 = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 1]);  mul_717 = None
    sum_231: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_234, [0, 1]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_20: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_710, mul_716], 2);  mul_710 = mul_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_719: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_65, 0.5);  add_65 = None
    mul_720: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, view_56)
    mul_721: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_720, -0.5);  mul_720 = None
    exp_20: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_721);  mul_721 = None
    mul_722: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_723: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_56, mul_722);  view_56 = mul_722 = None
    add_273: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_719, mul_723);  mul_719 = mul_723 = None
    mul_724: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_20, add_273);  cat_20 = add_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_368: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_724, [1568, 1536]);  mul_724 = None
    mm_156: "f32[1568, 256]" = torch.ops.aten.mm.default(view_368, permute_446);  permute_446 = None
    permute_447: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_157: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_447, view_55);  permute_447 = view_55 = None
    permute_448: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_232: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[1536]" = torch.ops.aten.view.default(sum_232, [1536]);  sum_232 = None
    permute_449: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_370: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_156, [8, 196, 256]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_726: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_370, primals_93);  primals_93 = None
    mul_727: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_726, 256)
    sum_233: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_726, [2], True)
    mul_728: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_726, mul_72);  mul_726 = None
    sum_234: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_728, [2], True);  mul_728 = None
    mul_729: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_72, sum_234);  sum_234 = None
    sub_188: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_727, sum_233);  mul_727 = sum_233 = None
    sub_189: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_188, mul_729);  sub_188 = mul_729 = None
    mul_730: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_189);  div_43 = sub_189 = None
    mul_731: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_370, mul_72);  mul_72 = None
    sum_235: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 1]);  mul_731 = None
    sum_236: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_370, [0, 1]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_274: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_271, mul_730);  add_271 = mul_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_371: "f32[1568, 256]" = torch.ops.aten.view.default(add_274, [1568, 256])
    mm_158: "f32[1568, 768]" = torch.ops.aten.mm.default(view_371, permute_450);  permute_450 = None
    permute_451: "f32[256, 1568]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_159: "f32[256, 768]" = torch.ops.aten.mm.default(permute_451, view_53);  permute_451 = view_53 = None
    permute_452: "f32[768, 256]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_237: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[256]" = torch.ops.aten.view.default(sum_237, [256]);  sum_237 = None
    permute_453: "f32[256, 768]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    view_373: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_158, [8, 196, 768]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_732: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_373, getitem_50);  getitem_50 = None
    mul_733: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_373, permute_44);  view_373 = permute_44 = None
    permute_454: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_732, [0, 2, 1]);  mul_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_238: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_454, [0, 1], True)
    view_374: "f32[196]" = torch.ops.aten.view.default(sum_238, [196]);  sum_238 = None
    clone_237: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_375: "f32[6144, 196]" = torch.ops.aten.view.default(clone_237, [6144, 196]);  clone_237 = None
    permute_455: "f32[196, 6144]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_160: "f32[196, 196]" = torch.ops.aten.mm.default(permute_455, view_51);  permute_455 = view_51 = None
    permute_456: "f32[196, 196]" = torch.ops.aten.permute.default(mm_160, [1, 0]);  mm_160 = None
    mm_161: "f32[6144, 196]" = torch.ops.aten.mm.default(view_375, permute_457);  view_375 = permute_457 = None
    view_376: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_161, [8, 768, 196]);  mm_161 = None
    permute_458: "f32[196, 196]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    permute_459: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_376, [0, 2, 1]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_238: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    mul_735: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_238, primals_87);  primals_87 = None
    mul_736: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_735, 768)
    sum_239: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_735, [2], True)
    mul_737: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_735, mul_69);  mul_735 = None
    sum_240: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_737, [2], True);  mul_737 = None
    mul_738: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_69, sum_240);  sum_240 = None
    sub_191: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_736, sum_239);  mul_736 = sum_239 = None
    sub_192: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_191, mul_738);  sub_191 = mul_738 = None
    mul_739: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_192);  div_44 = sub_192 = None
    mul_740: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_238, mul_69);  mul_69 = None
    sum_241: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_740, [0, 1]);  mul_740 = None
    sum_242: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_238, [0, 1]);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_21: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_733, mul_739], 2);  mul_733 = mul_739 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_742: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
    mul_743: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, view_50)
    mul_744: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_743, -0.5);  mul_743 = None
    exp_21: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_744);  mul_744 = None
    mul_745: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_746: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_50, mul_745);  view_50 = mul_745 = None
    add_276: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_742, mul_746);  mul_742 = mul_746 = None
    mul_747: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_21, add_276);  cat_21 = add_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_377: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_747, [1568, 1536]);  mul_747 = None
    mm_162: "f32[1568, 256]" = torch.ops.aten.mm.default(view_377, permute_460);  permute_460 = None
    permute_461: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_163: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_461, view_49);  permute_461 = view_49 = None
    permute_462: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[1536]" = torch.ops.aten.view.default(sum_243, [1536]);  sum_243 = None
    permute_463: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_379: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_162, [8, 196, 256]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_749: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_379, primals_83);  primals_83 = None
    mul_750: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_749, 256)
    sum_244: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True)
    mul_751: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_749, mul_64);  mul_749 = None
    sum_245: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_751, [2], True);  mul_751 = None
    mul_752: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_64, sum_245);  sum_245 = None
    sub_194: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_750, sum_244);  mul_750 = sum_244 = None
    sub_195: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_194, mul_752);  sub_194 = mul_752 = None
    mul_753: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_195);  div_45 = sub_195 = None
    mul_754: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_379, mul_64);  mul_64 = None
    sum_246: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 1]);  mul_754 = None
    sum_247: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_379, [0, 1]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_277: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_274, mul_753);  add_274 = mul_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_380: "f32[1568, 256]" = torch.ops.aten.view.default(add_277, [1568, 256])
    mm_164: "f32[1568, 768]" = torch.ops.aten.mm.default(view_380, permute_464);  permute_464 = None
    permute_465: "f32[256, 1568]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_165: "f32[256, 768]" = torch.ops.aten.mm.default(permute_465, view_47);  permute_465 = view_47 = None
    permute_466: "f32[768, 256]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_248: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[256]" = torch.ops.aten.view.default(sum_248, [256]);  sum_248 = None
    permute_467: "f32[256, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_382: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_164, [8, 196, 768]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_755: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_382, getitem_44);  getitem_44 = None
    mul_756: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_382, permute_39);  view_382 = permute_39 = None
    permute_468: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_755, [0, 2, 1]);  mul_755 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_249: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_468, [0, 1], True)
    view_383: "f32[196]" = torch.ops.aten.view.default(sum_249, [196]);  sum_249 = None
    clone_241: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
    view_384: "f32[6144, 196]" = torch.ops.aten.view.default(clone_241, [6144, 196]);  clone_241 = None
    permute_469: "f32[196, 6144]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_166: "f32[196, 196]" = torch.ops.aten.mm.default(permute_469, view_45);  permute_469 = view_45 = None
    permute_470: "f32[196, 196]" = torch.ops.aten.permute.default(mm_166, [1, 0]);  mm_166 = None
    mm_167: "f32[6144, 196]" = torch.ops.aten.mm.default(view_384, permute_471);  view_384 = permute_471 = None
    view_385: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_167, [8, 768, 196]);  mm_167 = None
    permute_472: "f32[196, 196]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    permute_473: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_242: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
    mul_758: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_242, primals_77);  primals_77 = None
    mul_759: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_758, 768)
    sum_250: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_758, [2], True)
    mul_760: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_758, mul_61);  mul_758 = None
    sum_251: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [2], True);  mul_760 = None
    mul_761: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_61, sum_251);  sum_251 = None
    sub_197: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_759, sum_250);  mul_759 = sum_250 = None
    sub_198: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_197, mul_761);  sub_197 = mul_761 = None
    mul_762: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_198);  div_46 = sub_198 = None
    mul_763: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_242, mul_61);  mul_61 = None
    sum_252: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 1]);  mul_763 = None
    sum_253: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_242, [0, 1]);  clone_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_22: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_756, mul_762], 2);  mul_756 = mul_762 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_765: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
    mul_766: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, view_44)
    mul_767: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_766, -0.5);  mul_766 = None
    exp_22: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_767);  mul_767 = None
    mul_768: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_769: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_44, mul_768);  view_44 = mul_768 = None
    add_279: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_765, mul_769);  mul_765 = mul_769 = None
    mul_770: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_22, add_279);  cat_22 = add_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_386: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_770, [1568, 1536]);  mul_770 = None
    mm_168: "f32[1568, 256]" = torch.ops.aten.mm.default(view_386, permute_474);  permute_474 = None
    permute_475: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_169: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_475, view_43);  permute_475 = view_43 = None
    permute_476: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_254: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[1536]" = torch.ops.aten.view.default(sum_254, [1536]);  sum_254 = None
    permute_477: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
    view_388: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_168, [8, 196, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_772: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_388, primals_73);  primals_73 = None
    mul_773: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_772, 256)
    sum_255: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_772, [2], True)
    mul_774: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_772, mul_56);  mul_772 = None
    sum_256: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_774, [2], True);  mul_774 = None
    mul_775: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_56, sum_256);  sum_256 = None
    sub_200: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_773, sum_255);  mul_773 = sum_255 = None
    sub_201: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_200, mul_775);  sub_200 = mul_775 = None
    mul_776: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_47, sub_201);  div_47 = sub_201 = None
    mul_777: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_388, mul_56);  mul_56 = None
    sum_257: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 1]);  mul_777 = None
    sum_258: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_388, [0, 1]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_280: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_277, mul_776);  add_277 = mul_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_389: "f32[1568, 256]" = torch.ops.aten.view.default(add_280, [1568, 256])
    mm_170: "f32[1568, 768]" = torch.ops.aten.mm.default(view_389, permute_478);  permute_478 = None
    permute_479: "f32[256, 1568]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_171: "f32[256, 768]" = torch.ops.aten.mm.default(permute_479, view_41);  permute_479 = view_41 = None
    permute_480: "f32[768, 256]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_259: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[256]" = torch.ops.aten.view.default(sum_259, [256]);  sum_259 = None
    permute_481: "f32[256, 768]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
    view_391: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_170, [8, 196, 768]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_778: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_391, getitem_38);  getitem_38 = None
    mul_779: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_391, permute_34);  view_391 = permute_34 = None
    permute_482: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_778, [0, 2, 1]);  mul_778 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_260: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_482, [0, 1], True)
    view_392: "f32[196]" = torch.ops.aten.view.default(sum_260, [196]);  sum_260 = None
    clone_245: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_482, memory_format = torch.contiguous_format);  permute_482 = None
    view_393: "f32[6144, 196]" = torch.ops.aten.view.default(clone_245, [6144, 196]);  clone_245 = None
    permute_483: "f32[196, 6144]" = torch.ops.aten.permute.default(view_393, [1, 0])
    mm_172: "f32[196, 196]" = torch.ops.aten.mm.default(permute_483, view_39);  permute_483 = view_39 = None
    permute_484: "f32[196, 196]" = torch.ops.aten.permute.default(mm_172, [1, 0]);  mm_172 = None
    mm_173: "f32[6144, 196]" = torch.ops.aten.mm.default(view_393, permute_485);  view_393 = permute_485 = None
    view_394: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_173, [8, 768, 196]);  mm_173 = None
    permute_486: "f32[196, 196]" = torch.ops.aten.permute.default(permute_484, [1, 0]);  permute_484 = None
    permute_487: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_394, [0, 2, 1]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_246: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    mul_781: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_246, primals_67);  primals_67 = None
    mul_782: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_781, 768)
    sum_261: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_781, [2], True)
    mul_783: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_781, mul_53);  mul_781 = None
    sum_262: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_783, [2], True);  mul_783 = None
    mul_784: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_53, sum_262);  sum_262 = None
    sub_203: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_782, sum_261);  mul_782 = sum_261 = None
    sub_204: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_203, mul_784);  sub_203 = mul_784 = None
    mul_785: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_204);  div_48 = sub_204 = None
    mul_786: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_246, mul_53);  mul_53 = None
    sum_263: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_786, [0, 1]);  mul_786 = None
    sum_264: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_246, [0, 1]);  clone_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_23: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_779, mul_785], 2);  mul_779 = mul_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_788: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
    mul_789: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_790: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_789, -0.5);  mul_789 = None
    exp_23: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_790);  mul_790 = None
    mul_791: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_792: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_38, mul_791);  view_38 = mul_791 = None
    add_282: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_788, mul_792);  mul_788 = mul_792 = None
    mul_793: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_23, add_282);  cat_23 = add_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_395: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_793, [1568, 1536]);  mul_793 = None
    mm_174: "f32[1568, 256]" = torch.ops.aten.mm.default(view_395, permute_488);  permute_488 = None
    permute_489: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_175: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_489, view_37);  permute_489 = view_37 = None
    permute_490: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_265: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[1536]" = torch.ops.aten.view.default(sum_265, [1536]);  sum_265 = None
    permute_491: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_397: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_174, [8, 196, 256]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_795: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_397, primals_63);  primals_63 = None
    mul_796: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_795, 256)
    sum_266: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_795, [2], True)
    mul_797: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_795, mul_48);  mul_795 = None
    sum_267: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_797, [2], True);  mul_797 = None
    mul_798: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_48, sum_267);  sum_267 = None
    sub_206: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_796, sum_266);  mul_796 = sum_266 = None
    sub_207: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_206, mul_798);  sub_206 = mul_798 = None
    mul_799: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_207);  div_49 = sub_207 = None
    mul_800: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_397, mul_48);  mul_48 = None
    sum_268: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_800, [0, 1]);  mul_800 = None
    sum_269: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_397, [0, 1]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_283: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_280, mul_799);  add_280 = mul_799 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_398: "f32[1568, 256]" = torch.ops.aten.view.default(add_283, [1568, 256])
    mm_176: "f32[1568, 768]" = torch.ops.aten.mm.default(view_398, permute_492);  permute_492 = None
    permute_493: "f32[256, 1568]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_177: "f32[256, 768]" = torch.ops.aten.mm.default(permute_493, view_35);  permute_493 = view_35 = None
    permute_494: "f32[768, 256]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_270: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
    view_399: "f32[256]" = torch.ops.aten.view.default(sum_270, [256]);  sum_270 = None
    permute_495: "f32[256, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_400: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_176, [8, 196, 768]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_801: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, getitem_32);  getitem_32 = None
    mul_802: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_400, permute_29);  view_400 = permute_29 = None
    permute_496: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_801, [0, 2, 1]);  mul_801 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_271: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_496, [0, 1], True)
    view_401: "f32[196]" = torch.ops.aten.view.default(sum_271, [196]);  sum_271 = None
    clone_249: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_496, memory_format = torch.contiguous_format);  permute_496 = None
    view_402: "f32[6144, 196]" = torch.ops.aten.view.default(clone_249, [6144, 196]);  clone_249 = None
    permute_497: "f32[196, 6144]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_178: "f32[196, 196]" = torch.ops.aten.mm.default(permute_497, view_33);  permute_497 = view_33 = None
    permute_498: "f32[196, 196]" = torch.ops.aten.permute.default(mm_178, [1, 0]);  mm_178 = None
    mm_179: "f32[6144, 196]" = torch.ops.aten.mm.default(view_402, permute_499);  view_402 = permute_499 = None
    view_403: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_179, [8, 768, 196]);  mm_179 = None
    permute_500: "f32[196, 196]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    permute_501: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_250: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
    mul_804: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_250, primals_57);  primals_57 = None
    mul_805: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_804, 768)
    sum_272: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True)
    mul_806: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_804, mul_45);  mul_804 = None
    sum_273: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [2], True);  mul_806 = None
    mul_807: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_273);  sum_273 = None
    sub_209: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_805, sum_272);  mul_805 = sum_272 = None
    sub_210: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_209, mul_807);  sub_209 = mul_807 = None
    mul_808: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_210);  div_50 = sub_210 = None
    mul_809: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_250, mul_45);  mul_45 = None
    sum_274: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_809, [0, 1]);  mul_809 = None
    sum_275: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_250, [0, 1]);  clone_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_24: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_802, mul_808], 2);  mul_802 = mul_808 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_811: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_37, 0.5);  add_37 = None
    mul_812: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, view_32)
    mul_813: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_812, -0.5);  mul_812 = None
    exp_24: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_813);  mul_813 = None
    mul_814: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_815: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_32, mul_814);  view_32 = mul_814 = None
    add_285: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_811, mul_815);  mul_811 = mul_815 = None
    mul_816: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_24, add_285);  cat_24 = add_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_404: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_816, [1568, 1536]);  mul_816 = None
    mm_180: "f32[1568, 256]" = torch.ops.aten.mm.default(view_404, permute_502);  permute_502 = None
    permute_503: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_181: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_503, view_31);  permute_503 = view_31 = None
    permute_504: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_181, [1, 0]);  mm_181 = None
    sum_276: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[1536]" = torch.ops.aten.view.default(sum_276, [1536]);  sum_276 = None
    permute_505: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_406: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_180, [8, 196, 256]);  mm_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_818: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_406, primals_53);  primals_53 = None
    mul_819: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_818, 256)
    sum_277: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_818, [2], True)
    mul_820: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_818, mul_40);  mul_818 = None
    sum_278: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_820, [2], True);  mul_820 = None
    mul_821: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_40, sum_278);  sum_278 = None
    sub_212: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_819, sum_277);  mul_819 = sum_277 = None
    sub_213: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_212, mul_821);  sub_212 = mul_821 = None
    mul_822: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_213);  div_51 = sub_213 = None
    mul_823: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_406, mul_40);  mul_40 = None
    sum_279: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 1]);  mul_823 = None
    sum_280: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_406, [0, 1]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_286: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_283, mul_822);  add_283 = mul_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_407: "f32[1568, 256]" = torch.ops.aten.view.default(add_286, [1568, 256])
    mm_182: "f32[1568, 768]" = torch.ops.aten.mm.default(view_407, permute_506);  permute_506 = None
    permute_507: "f32[256, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_183: "f32[256, 768]" = torch.ops.aten.mm.default(permute_507, view_29);  permute_507 = view_29 = None
    permute_508: "f32[768, 256]" = torch.ops.aten.permute.default(mm_183, [1, 0]);  mm_183 = None
    sum_281: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[256]" = torch.ops.aten.view.default(sum_281, [256]);  sum_281 = None
    permute_509: "f32[256, 768]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_409: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_182, [8, 196, 768]);  mm_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_824: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_409, getitem_26);  getitem_26 = None
    mul_825: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_409, permute_24);  view_409 = permute_24 = None
    permute_510: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_824, [0, 2, 1]);  mul_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_282: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_510, [0, 1], True)
    view_410: "f32[196]" = torch.ops.aten.view.default(sum_282, [196]);  sum_282 = None
    clone_253: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_510, memory_format = torch.contiguous_format);  permute_510 = None
    view_411: "f32[6144, 196]" = torch.ops.aten.view.default(clone_253, [6144, 196]);  clone_253 = None
    permute_511: "f32[196, 6144]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_184: "f32[196, 196]" = torch.ops.aten.mm.default(permute_511, view_27);  permute_511 = view_27 = None
    permute_512: "f32[196, 196]" = torch.ops.aten.permute.default(mm_184, [1, 0]);  mm_184 = None
    mm_185: "f32[6144, 196]" = torch.ops.aten.mm.default(view_411, permute_513);  view_411 = permute_513 = None
    view_412: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_185, [8, 768, 196]);  mm_185 = None
    permute_514: "f32[196, 196]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    permute_515: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_412, [0, 2, 1]);  view_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_254: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_515, memory_format = torch.contiguous_format);  permute_515 = None
    mul_827: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_254, primals_47);  primals_47 = None
    mul_828: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_827, 768)
    sum_283: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_827, [2], True)
    mul_829: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_827, mul_37);  mul_827 = None
    sum_284: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_829, [2], True);  mul_829 = None
    mul_830: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_284);  sum_284 = None
    sub_215: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_828, sum_283);  mul_828 = sum_283 = None
    sub_216: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_215, mul_830);  sub_215 = mul_830 = None
    mul_831: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_216);  div_52 = sub_216 = None
    mul_832: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_254, mul_37);  mul_37 = None
    sum_285: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 1]);  mul_832 = None
    sum_286: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_254, [0, 1]);  clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_25: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_825, mul_831], 2);  mul_825 = mul_831 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_834: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_835: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, view_26)
    mul_836: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_835, -0.5);  mul_835 = None
    exp_25: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_836);  mul_836 = None
    mul_837: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_838: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_26, mul_837);  view_26 = mul_837 = None
    add_288: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_834, mul_838);  mul_834 = mul_838 = None
    mul_839: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_25, add_288);  cat_25 = add_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_413: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_839, [1568, 1536]);  mul_839 = None
    mm_186: "f32[1568, 256]" = torch.ops.aten.mm.default(view_413, permute_516);  permute_516 = None
    permute_517: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_187: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_517, view_25);  permute_517 = view_25 = None
    permute_518: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_187, [1, 0]);  mm_187 = None
    sum_287: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[1536]" = torch.ops.aten.view.default(sum_287, [1536]);  sum_287 = None
    permute_519: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_518, [1, 0]);  permute_518 = None
    view_415: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_186, [8, 196, 256]);  mm_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_841: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_415, primals_43);  primals_43 = None
    mul_842: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_841, 256)
    sum_288: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_841, [2], True)
    mul_843: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_841, mul_32);  mul_841 = None
    sum_289: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True);  mul_843 = None
    mul_844: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_32, sum_289);  sum_289 = None
    sub_218: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_842, sum_288);  mul_842 = sum_288 = None
    sub_219: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_218, mul_844);  sub_218 = mul_844 = None
    mul_845: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_53, sub_219);  div_53 = sub_219 = None
    mul_846: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_415, mul_32);  mul_32 = None
    sum_290: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_846, [0, 1]);  mul_846 = None
    sum_291: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_415, [0, 1]);  view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_289: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_286, mul_845);  add_286 = mul_845 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_416: "f32[1568, 256]" = torch.ops.aten.view.default(add_289, [1568, 256])
    mm_188: "f32[1568, 768]" = torch.ops.aten.mm.default(view_416, permute_520);  permute_520 = None
    permute_521: "f32[256, 1568]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_189: "f32[256, 768]" = torch.ops.aten.mm.default(permute_521, view_23);  permute_521 = view_23 = None
    permute_522: "f32[768, 256]" = torch.ops.aten.permute.default(mm_189, [1, 0]);  mm_189 = None
    sum_292: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.view.default(sum_292, [256]);  sum_292 = None
    permute_523: "f32[256, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_418: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_188, [8, 196, 768]);  mm_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_847: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_418, getitem_20);  getitem_20 = None
    mul_848: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_418, permute_19);  view_418 = permute_19 = None
    permute_524: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_847, [0, 2, 1]);  mul_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_293: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_524, [0, 1], True)
    view_419: "f32[196]" = torch.ops.aten.view.default(sum_293, [196]);  sum_293 = None
    clone_257: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_420: "f32[6144, 196]" = torch.ops.aten.view.default(clone_257, [6144, 196]);  clone_257 = None
    permute_525: "f32[196, 6144]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_190: "f32[196, 196]" = torch.ops.aten.mm.default(permute_525, view_21);  permute_525 = view_21 = None
    permute_526: "f32[196, 196]" = torch.ops.aten.permute.default(mm_190, [1, 0]);  mm_190 = None
    mm_191: "f32[6144, 196]" = torch.ops.aten.mm.default(view_420, permute_527);  view_420 = permute_527 = None
    view_421: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_191, [8, 768, 196]);  mm_191 = None
    permute_528: "f32[196, 196]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    permute_529: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_421, [0, 2, 1]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_258: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_529, memory_format = torch.contiguous_format);  permute_529 = None
    mul_850: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_258, primals_37);  primals_37 = None
    mul_851: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_850, 768)
    sum_294: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_850, [2], True)
    mul_852: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_850, mul_29);  mul_850 = None
    sum_295: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_852, [2], True);  mul_852 = None
    mul_853: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_295);  sum_295 = None
    sub_221: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_851, sum_294);  mul_851 = sum_294 = None
    sub_222: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_221, mul_853);  sub_221 = mul_853 = None
    mul_854: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_222);  div_54 = sub_222 = None
    mul_855: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_258, mul_29);  mul_29 = None
    sum_296: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_855, [0, 1]);  mul_855 = None
    sum_297: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_258, [0, 1]);  clone_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_26: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_848, mul_854], 2);  mul_848 = mul_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_857: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_23, 0.5);  add_23 = None
    mul_858: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, view_20)
    mul_859: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_858, -0.5);  mul_858 = None
    exp_26: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_859);  mul_859 = None
    mul_860: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_861: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_20, mul_860);  view_20 = mul_860 = None
    add_291: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_857, mul_861);  mul_857 = mul_861 = None
    mul_862: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_26, add_291);  cat_26 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_422: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_862, [1568, 1536]);  mul_862 = None
    mm_192: "f32[1568, 256]" = torch.ops.aten.mm.default(view_422, permute_530);  permute_530 = None
    permute_531: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_193: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_531, view_19);  permute_531 = view_19 = None
    permute_532: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_193, [1, 0]);  mm_193 = None
    sum_298: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[1536]" = torch.ops.aten.view.default(sum_298, [1536]);  sum_298 = None
    permute_533: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_424: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_192, [8, 196, 256]);  mm_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_864: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_424, primals_33);  primals_33 = None
    mul_865: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_864, 256)
    sum_299: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_864, [2], True)
    mul_866: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_864, mul_24);  mul_864 = None
    sum_300: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_866, [2], True);  mul_866 = None
    mul_867: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_24, sum_300);  sum_300 = None
    sub_224: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_865, sum_299);  mul_865 = sum_299 = None
    sub_225: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_224, mul_867);  sub_224 = mul_867 = None
    mul_868: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_225);  div_55 = sub_225 = None
    mul_869: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_424, mul_24);  mul_24 = None
    sum_301: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 1]);  mul_869 = None
    sum_302: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_424, [0, 1]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_292: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_289, mul_868);  add_289 = mul_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_425: "f32[1568, 256]" = torch.ops.aten.view.default(add_292, [1568, 256])
    mm_194: "f32[1568, 768]" = torch.ops.aten.mm.default(view_425, permute_534);  permute_534 = None
    permute_535: "f32[256, 1568]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_195: "f32[256, 768]" = torch.ops.aten.mm.default(permute_535, view_17);  permute_535 = view_17 = None
    permute_536: "f32[768, 256]" = torch.ops.aten.permute.default(mm_195, [1, 0]);  mm_195 = None
    sum_303: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[256]" = torch.ops.aten.view.default(sum_303, [256]);  sum_303 = None
    permute_537: "f32[256, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_427: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_194, [8, 196, 768]);  mm_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_870: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_427, getitem_14);  getitem_14 = None
    mul_871: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_427, permute_14);  view_427 = permute_14 = None
    permute_538: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_870, [0, 2, 1]);  mul_870 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_304: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_538, [0, 1], True)
    view_428: "f32[196]" = torch.ops.aten.view.default(sum_304, [196]);  sum_304 = None
    clone_261: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_538, memory_format = torch.contiguous_format);  permute_538 = None
    view_429: "f32[6144, 196]" = torch.ops.aten.view.default(clone_261, [6144, 196]);  clone_261 = None
    permute_539: "f32[196, 6144]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_196: "f32[196, 196]" = torch.ops.aten.mm.default(permute_539, view_15);  permute_539 = view_15 = None
    permute_540: "f32[196, 196]" = torch.ops.aten.permute.default(mm_196, [1, 0]);  mm_196 = None
    mm_197: "f32[6144, 196]" = torch.ops.aten.mm.default(view_429, permute_541);  view_429 = permute_541 = None
    view_430: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_197, [8, 768, 196]);  mm_197 = None
    permute_542: "f32[196, 196]" = torch.ops.aten.permute.default(permute_540, [1, 0]);  permute_540 = None
    permute_543: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_430, [0, 2, 1]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_262: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_543, memory_format = torch.contiguous_format);  permute_543 = None
    mul_873: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_262, primals_27);  primals_27 = None
    mul_874: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_873, 768)
    sum_305: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True)
    mul_875: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_873, mul_21);  mul_873 = None
    sum_306: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_875, [2], True);  mul_875 = None
    mul_876: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_306);  sum_306 = None
    sub_227: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_874, sum_305);  mul_874 = sum_305 = None
    sub_228: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_227, mul_876);  sub_227 = mul_876 = None
    mul_877: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_228);  div_56 = sub_228 = None
    mul_878: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_262, mul_21);  mul_21 = None
    sum_307: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 1]);  mul_878 = None
    sum_308: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_262, [0, 1]);  clone_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_27: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_871, mul_877], 2);  mul_871 = mul_877 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_880: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_881: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, view_14)
    mul_882: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_881, -0.5);  mul_881 = None
    exp_27: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_882);  mul_882 = None
    mul_883: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_884: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_14, mul_883);  view_14 = mul_883 = None
    add_294: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_880, mul_884);  mul_880 = mul_884 = None
    mul_885: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_27, add_294);  cat_27 = add_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_431: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_885, [1568, 1536]);  mul_885 = None
    mm_198: "f32[1568, 256]" = torch.ops.aten.mm.default(view_431, permute_544);  permute_544 = None
    permute_545: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_199: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_545, view_13);  permute_545 = view_13 = None
    permute_546: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_199, [1, 0]);  mm_199 = None
    sum_309: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[1536]" = torch.ops.aten.view.default(sum_309, [1536]);  sum_309 = None
    permute_547: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_546, [1, 0]);  permute_546 = None
    view_433: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_198, [8, 196, 256]);  mm_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_887: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_433, primals_23);  primals_23 = None
    mul_888: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_887, 256)
    sum_310: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
    mul_889: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_887, mul_16);  mul_887 = None
    sum_311: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
    mul_890: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_16, sum_311);  sum_311 = None
    sub_230: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_888, sum_310);  mul_888 = sum_310 = None
    sub_231: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_230, mul_890);  sub_230 = mul_890 = None
    mul_891: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_231);  div_57 = sub_231 = None
    mul_892: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_433, mul_16);  mul_16 = None
    sum_312: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
    sum_313: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_295: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_292, mul_891);  add_292 = mul_891 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_434: "f32[1568, 256]" = torch.ops.aten.view.default(add_295, [1568, 256])
    mm_200: "f32[1568, 768]" = torch.ops.aten.mm.default(view_434, permute_548);  permute_548 = None
    permute_549: "f32[256, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_201: "f32[256, 768]" = torch.ops.aten.mm.default(permute_549, view_11);  permute_549 = view_11 = None
    permute_550: "f32[768, 256]" = torch.ops.aten.permute.default(mm_201, [1, 0]);  mm_201 = None
    sum_314: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[256]" = torch.ops.aten.view.default(sum_314, [256]);  sum_314 = None
    permute_551: "f32[256, 768]" = torch.ops.aten.permute.default(permute_550, [1, 0]);  permute_550 = None
    view_436: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_200, [8, 196, 768]);  mm_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_893: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, getitem_8);  getitem_8 = None
    mul_894: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_436, permute_9);  view_436 = permute_9 = None
    permute_552: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_893, [0, 2, 1]);  mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_315: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_552, [0, 1], True)
    view_437: "f32[196]" = torch.ops.aten.view.default(sum_315, [196]);  sum_315 = None
    clone_265: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_552, memory_format = torch.contiguous_format);  permute_552 = None
    view_438: "f32[6144, 196]" = torch.ops.aten.view.default(clone_265, [6144, 196]);  clone_265 = None
    permute_553: "f32[196, 6144]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_202: "f32[196, 196]" = torch.ops.aten.mm.default(permute_553, view_9);  permute_553 = view_9 = None
    permute_554: "f32[196, 196]" = torch.ops.aten.permute.default(mm_202, [1, 0]);  mm_202 = None
    mm_203: "f32[6144, 196]" = torch.ops.aten.mm.default(view_438, permute_555);  view_438 = permute_555 = None
    view_439: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_203, [8, 768, 196]);  mm_203 = None
    permute_556: "f32[196, 196]" = torch.ops.aten.permute.default(permute_554, [1, 0]);  permute_554 = None
    permute_557: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_266: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_557, memory_format = torch.contiguous_format);  permute_557 = None
    mul_896: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_266, primals_17);  primals_17 = None
    mul_897: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_896, 768)
    sum_316: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
    mul_898: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_896, mul_13);  mul_896 = None
    sum_317: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
    mul_899: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_13, sum_317);  sum_317 = None
    sub_233: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_897, sum_316);  mul_897 = sum_316 = None
    sub_234: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_233, mul_899);  sub_233 = mul_899 = None
    mul_900: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_234);  div_58 = sub_234 = None
    mul_901: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_266, mul_13);  mul_13 = None
    sum_318: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
    sum_319: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_266, [0, 1]);  clone_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_28: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_894, mul_900], 2);  mul_894 = mul_900 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_903: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
    mul_904: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_905: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_904, -0.5);  mul_904 = None
    exp_28: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_905);  mul_905 = None
    mul_906: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_907: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_8, mul_906);  view_8 = mul_906 = None
    add_297: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_903, mul_907);  mul_903 = mul_907 = None
    mul_908: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_28, add_297);  cat_28 = add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_440: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_908, [1568, 1536]);  mul_908 = None
    mm_204: "f32[1568, 256]" = torch.ops.aten.mm.default(view_440, permute_558);  permute_558 = None
    permute_559: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_205: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_559, view_7);  permute_559 = view_7 = None
    permute_560: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_205, [1, 0]);  mm_205 = None
    sum_320: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[1536]" = torch.ops.aten.view.default(sum_320, [1536]);  sum_320 = None
    permute_561: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_560, [1, 0]);  permute_560 = None
    view_442: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_204, [8, 196, 256]);  mm_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_910: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_442, primals_13);  primals_13 = None
    mul_911: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_910, 256)
    sum_321: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_910, [2], True)
    mul_912: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_910, mul_8);  mul_910 = None
    sum_322: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True);  mul_912 = None
    mul_913: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_8, sum_322);  sum_322 = None
    sub_236: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_911, sum_321);  mul_911 = sum_321 = None
    sub_237: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_236, mul_913);  sub_236 = mul_913 = None
    mul_914: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_59, sub_237);  div_59 = sub_237 = None
    mul_915: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_442, mul_8);  mul_8 = None
    sum_323: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_915, [0, 1]);  mul_915 = None
    sum_324: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_442, [0, 1]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_298: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_295, mul_914);  add_295 = mul_914 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:190, code: x = self.fc2(x)
    view_443: "f32[1568, 256]" = torch.ops.aten.view.default(add_298, [1568, 256])
    mm_206: "f32[1568, 768]" = torch.ops.aten.mm.default(view_443, permute_562);  permute_562 = None
    permute_563: "f32[256, 1568]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_207: "f32[256, 768]" = torch.ops.aten.mm.default(permute_563, view_5);  permute_563 = view_5 = None
    permute_564: "f32[768, 256]" = torch.ops.aten.permute.default(mm_207, [1, 0]);  mm_207 = None
    sum_325: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[256]" = torch.ops.aten.view.default(sum_325, [256]);  sum_325 = None
    permute_565: "f32[256, 768]" = torch.ops.aten.permute.default(permute_564, [1, 0]);  permute_564 = None
    view_445: "f32[8, 196, 768]" = torch.ops.aten.view.default(mm_206, [8, 196, 768]);  mm_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:148, code: return u * v.transpose(-1, -2)
    mul_916: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_445, getitem_2);  getitem_2 = None
    mul_917: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_445, permute_4);  view_445 = permute_4 = None
    permute_566: "f32[8, 768, 196]" = torch.ops.aten.permute.default(mul_916, [0, 2, 1]);  mul_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:147, code: v = self.proj(v.transpose(-1, -2))
    sum_326: "f32[1, 1, 196]" = torch.ops.aten.sum.dim_IntList(permute_566, [0, 1], True)
    view_446: "f32[196]" = torch.ops.aten.view.default(sum_326, [196]);  sum_326 = None
    clone_269: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
    view_447: "f32[6144, 196]" = torch.ops.aten.view.default(clone_269, [6144, 196]);  clone_269 = None
    permute_567: "f32[196, 6144]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_208: "f32[196, 196]" = torch.ops.aten.mm.default(permute_567, view_3);  permute_567 = view_3 = None
    permute_568: "f32[196, 196]" = torch.ops.aten.permute.default(mm_208, [1, 0]);  mm_208 = None
    mm_209: "f32[6144, 196]" = torch.ops.aten.mm.default(view_447, permute_569);  view_447 = permute_569 = None
    view_448: "f32[8, 768, 196]" = torch.ops.aten.view.default(mm_209, [8, 768, 196]);  mm_209 = None
    permute_570: "f32[196, 196]" = torch.ops.aten.permute.default(permute_568, [1, 0]);  permute_568 = None
    permute_571: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:146, code: v = self.norm(v)
    clone_270: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_571, memory_format = torch.contiguous_format);  permute_571 = None
    mul_919: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_270, primals_7);  primals_7 = None
    mul_920: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_919, 768)
    sum_327: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_919, [2], True)
    mul_921: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_919, mul_5);  mul_919 = None
    sum_328: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_921, [2], True);  mul_921 = None
    mul_922: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, sum_328);  sum_328 = None
    sub_239: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_920, sum_327);  mul_920 = sum_327 = None
    sub_240: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_239, mul_922);  sub_239 = mul_922 = None
    mul_923: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_240);  div_60 = sub_240 = None
    mul_924: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_270, mul_5);  mul_5 = None
    sum_329: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 1]);  mul_924 = None
    sum_330: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_270, [0, 1]);  clone_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:145, code: u, v = x.chunk(2, dim=-1)
    cat_29: "f32[8, 196, 1536]" = torch.ops.aten.cat.default([mul_917, mul_923], 2);  mul_917 = mul_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:186, code: x = self.act(x)
    mul_926: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(add_2, 0.5);  add_2 = None
    mul_927: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, view_2)
    mul_928: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(mul_927, -0.5);  mul_927 = None
    exp_29: "f32[8, 196, 1536]" = torch.ops.aten.exp.default(mul_928);  mul_928 = None
    mul_929: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_930: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(view_2, mul_929);  view_2 = mul_929 = None
    add_300: "f32[8, 196, 1536]" = torch.ops.aten.add.Tensor(mul_926, mul_930);  mul_926 = mul_930 = None
    mul_931: "f32[8, 196, 1536]" = torch.ops.aten.mul.Tensor(cat_29, add_300);  cat_29 = add_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:185, code: x = self.fc1(x)
    view_449: "f32[1568, 1536]" = torch.ops.aten.view.default(mul_931, [1568, 1536]);  mul_931 = None
    mm_210: "f32[1568, 256]" = torch.ops.aten.mm.default(view_449, permute_572);  permute_572 = None
    permute_573: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_449, [1, 0])
    mm_211: "f32[1536, 256]" = torch.ops.aten.mm.default(permute_573, view_1);  permute_573 = view_1 = None
    permute_574: "f32[256, 1536]" = torch.ops.aten.permute.default(mm_211, [1, 0]);  mm_211 = None
    sum_331: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
    view_450: "f32[1536]" = torch.ops.aten.view.default(sum_331, [1536]);  sum_331 = None
    permute_575: "f32[1536, 256]" = torch.ops.aten.permute.default(permute_574, [1, 0]);  permute_574 = None
    view_451: "f32[8, 196, 256]" = torch.ops.aten.view.default(mm_210, [8, 196, 256]);  mm_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    mul_933: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_451, primals_3);  primals_3 = None
    mul_934: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_933, 256)
    sum_332: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_933, [2], True)
    mul_935: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul_933, mul);  mul_933 = None
    sum_333: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_935, [2], True);  mul_935 = None
    mul_936: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(mul, sum_333);  sum_333 = None
    sub_242: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(mul_934, sum_332);  mul_934 = sum_332 = None
    sub_243: "f32[8, 196, 256]" = torch.ops.aten.sub.Tensor(sub_242, mul_936);  sub_242 = mul_936 = None
    mul_937: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_243);  div_61 = sub_243 = None
    mul_938: "f32[8, 196, 256]" = torch.ops.aten.mul.Tensor(view_451, mul);  mul = None
    sum_334: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_938, [0, 1]);  mul_938 = None
    sum_335: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_451, [0, 1]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:175, code: x = x + self.drop_path(self.mlp_channels(self.norm(x)))
    add_301: "f32[8, 196, 256]" = torch.ops.aten.add.Tensor(add_298, mul_937);  add_298 = mul_937 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_576: "f32[8, 256, 196]" = torch.ops.aten.permute.default(add_301, [0, 2, 1]);  add_301 = None
    view_452: "f32[8, 256, 14, 14]" = torch.ops.aten.view.default(permute_576, [8, 256, 14, 14]);  permute_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_336: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_452, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_452, primals_307, primals_1, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_452 = primals_307 = primals_1 = None
    getitem_183: "f32[256, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return [getitem_183, sum_336, sum_334, sum_335, permute_575, view_450, sum_329, sum_330, permute_570, view_446, permute_565, view_444, sum_323, sum_324, permute_561, view_441, sum_318, sum_319, permute_556, view_437, permute_551, view_435, sum_312, sum_313, permute_547, view_432, sum_307, sum_308, permute_542, view_428, permute_537, view_426, sum_301, sum_302, permute_533, view_423, sum_296, sum_297, permute_528, view_419, permute_523, view_417, sum_290, sum_291, permute_519, view_414, sum_285, sum_286, permute_514, view_410, permute_509, view_408, sum_279, sum_280, permute_505, view_405, sum_274, sum_275, permute_500, view_401, permute_495, view_399, sum_268, sum_269, permute_491, view_396, sum_263, sum_264, permute_486, view_392, permute_481, view_390, sum_257, sum_258, permute_477, view_387, sum_252, sum_253, permute_472, view_383, permute_467, view_381, sum_246, sum_247, permute_463, view_378, sum_241, sum_242, permute_458, view_374, permute_453, view_372, sum_235, sum_236, permute_449, view_369, sum_230, sum_231, permute_444, view_365, permute_439, view_363, sum_224, sum_225, permute_435, view_360, sum_219, sum_220, permute_430, view_356, permute_425, view_354, sum_213, sum_214, permute_421, view_351, sum_208, sum_209, permute_416, view_347, permute_411, view_345, sum_202, sum_203, permute_407, view_342, sum_197, sum_198, permute_402, view_338, permute_397, view_336, sum_191, sum_192, permute_393, view_333, sum_186, sum_187, permute_388, view_329, permute_383, view_327, sum_180, sum_181, permute_379, view_324, sum_175, sum_176, permute_374, view_320, permute_369, view_318, sum_169, sum_170, permute_365, view_315, sum_164, sum_165, permute_360, view_311, permute_355, view_309, sum_158, sum_159, permute_351, view_306, sum_153, sum_154, permute_346, view_302, permute_341, view_300, sum_147, sum_148, permute_337, view_297, sum_142, sum_143, permute_332, view_293, permute_327, view_291, sum_136, sum_137, permute_323, view_288, sum_131, sum_132, permute_318, view_284, permute_313, view_282, sum_125, sum_126, permute_309, view_279, sum_120, sum_121, permute_304, view_275, permute_299, view_273, sum_114, sum_115, permute_295, view_270, sum_109, sum_110, permute_290, view_266, permute_285, view_264, sum_103, sum_104, permute_281, view_261, sum_98, sum_99, permute_276, view_257, permute_271, view_255, sum_92, sum_93, permute_267, view_252, sum_87, sum_88, permute_262, view_248, permute_257, view_246, sum_81, sum_82, permute_253, view_243, sum_76, sum_77, permute_248, view_239, permute_243, view_237, sum_70, sum_71, permute_239, view_234, sum_65, sum_66, permute_234, view_230, permute_229, view_228, sum_59, sum_60, permute_225, view_225, sum_54, sum_55, permute_220, view_221, permute_215, view_219, sum_48, sum_49, permute_211, view_216, sum_43, sum_44, permute_206, view_212, permute_201, view_210, sum_37, sum_38, permute_197, view_207, sum_32, sum_33, permute_192, view_203, permute_187, view_201, sum_26, sum_27, permute_183, view_198, sum_21, sum_22, permute_178, view_194, permute_173, view_192, sum_15, sum_16, permute_169, view_189, sum_10, sum_11, permute_164, view_185, permute_159, view_183, sum_4, sum_5, permute_155, view_181, None]
    