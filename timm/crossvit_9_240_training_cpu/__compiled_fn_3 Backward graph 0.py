from __future__ import annotations



def forward(self, primals_5: "f32[128, 3, 12, 12]", primals_7: "f32[256, 3, 16, 16]", primals_9: "f32[128]", primals_15: "f32[128]", primals_21: "f32[256]", primals_27: "f32[256]", primals_33: "f32[256]", primals_39: "f32[256]", primals_45: "f32[256]", primals_51: "f32[256]", primals_57: "f32[128]", primals_58: "f32[128]", primals_61: "f32[256]", primals_62: "f32[256]", primals_65: "f32[256]", primals_75: "f32[256]", primals_76: "f32[256]", primals_79: "f32[128]", primals_89: "f32[128]", primals_90: "f32[128]", primals_93: "f32[128]", primals_99: "f32[128]", primals_105: "f32[256]", primals_111: "f32[256]", primals_117: "f32[256]", primals_123: "f32[256]", primals_129: "f32[256]", primals_135: "f32[256]", primals_141: "f32[128]", primals_142: "f32[128]", primals_145: "f32[256]", primals_146: "f32[256]", primals_149: "f32[256]", primals_159: "f32[256]", primals_160: "f32[256]", primals_163: "f32[128]", primals_173: "f32[128]", primals_174: "f32[128]", primals_177: "f32[128]", primals_183: "f32[128]", primals_189: "f32[256]", primals_195: "f32[256]", primals_201: "f32[256]", primals_207: "f32[256]", primals_213: "f32[256]", primals_219: "f32[256]", primals_225: "f32[128]", primals_226: "f32[128]", primals_229: "f32[256]", primals_230: "f32[256]", primals_233: "f32[256]", primals_243: "f32[256]", primals_244: "f32[256]", primals_247: "f32[128]", primals_257: "f32[128]", primals_258: "f32[128]", primals_261: "f32[128]", primals_263: "f32[256]", primals_269: "f32[8, 3, 240, 240]", add_46: "f32[8, 3, 224, 224]", mul_82: "f32[8, 401, 128]", view_6: "f32[3208, 128]", getitem_2: "f32[8, 4, 401, 32]", getitem_3: "f32[8, 4, 401, 32]", getitem_4: "f32[8, 4, 401, 32]", getitem_6: "f32[8, 4, 401]", getitem_7: "i32[]", getitem_8: "i32[]", getitem_11: "i64[]", getitem_12: "i64[]", view_10: "f32[3208, 128]", mul_84: "f32[8, 401, 128]", view_12: "f32[3208, 128]", addmm_2: "f32[3208, 384]", view_14: "f32[3208, 384]", mul_89: "f32[8, 197, 256]", view_16: "f32[1576, 256]", getitem_18: "f32[8, 4, 197, 64]", getitem_19: "f32[8, 4, 197, 64]", getitem_20: "f32[8, 4, 197, 64]", getitem_22: "f32[8, 4, 197]", getitem_23: "i32[]", getitem_24: "i32[]", getitem_27: "i64[]", getitem_28: "i64[]", view_20: "f32[1576, 256]", mul_91: "f32[8, 197, 256]", view_22: "f32[1576, 256]", addmm_6: "f32[1576, 768]", view_24: "f32[1576, 768]", mul_96: "f32[8, 197, 256]", view_26: "f32[1576, 256]", getitem_34: "f32[8, 4, 197, 64]", getitem_35: "f32[8, 4, 197, 64]", getitem_36: "f32[8, 4, 197, 64]", getitem_38: "f32[8, 4, 197]", getitem_39: "i32[]", getitem_40: "i32[]", getitem_43: "i64[]", getitem_44: "i64[]", view_30: "f32[1576, 256]", mul_98: "f32[8, 197, 256]", view_32: "f32[1576, 256]", addmm_10: "f32[1576, 768]", view_34: "f32[1576, 768]", mul_103: "f32[8, 197, 256]", view_36: "f32[1576, 256]", getitem_50: "f32[8, 4, 197, 64]", getitem_51: "f32[8, 4, 197, 64]", getitem_52: "f32[8, 4, 197, 64]", getitem_54: "f32[8, 4, 197]", getitem_55: "i32[]", getitem_56: "i32[]", getitem_59: "i64[]", getitem_60: "i64[]", view_40: "f32[1576, 256]", mul_105: "f32[8, 197, 256]", view_42: "f32[1576, 256]", addmm_14: "f32[1576, 768]", view_44: "f32[1576, 768]", mul_110: "f32[8, 1, 128]", view_46: "f32[8, 128]", mul_115: "f32[8, 1, 256]", view_48: "f32[8, 256]", cat_2: "f32[8, 197, 256]", getitem_69: "f32[8, 197, 1]", rsqrt_10: "f32[8, 197, 1]", view_50: "f32[8, 256]", view_53: "f32[1576, 256]", view_66: "f32[8, 256]", mul_123: "f32[8, 1, 256]", view_68: "f32[8, 256]", cat_3: "f32[8, 401, 128]", cat_4: "f32[8, 401, 128]", getitem_73: "f32[8, 401, 1]", rsqrt_12: "f32[8, 401, 1]", view_70: "f32[8, 128]", view_73: "f32[3208, 128]", view_86: "f32[8, 128]", mul_131: "f32[8, 1, 128]", view_88: "f32[8, 128]", cat_5: "f32[8, 197, 256]", getitem_77: "f32[8, 401, 1]", rsqrt_14: "f32[8, 401, 1]", view_90: "f32[3208, 128]", getitem_78: "f32[8, 4, 401, 32]", getitem_79: "f32[8, 4, 401, 32]", getitem_80: "f32[8, 4, 401, 32]", getitem_82: "f32[8, 4, 401]", getitem_83: "i32[]", getitem_84: "i32[]", getitem_87: "i64[]", getitem_88: "i64[]", view_94: "f32[3208, 128]", mul_138: "f32[8, 401, 128]", view_96: "f32[3208, 128]", addmm_28: "f32[3208, 384]", view_98: "f32[3208, 384]", getitem_93: "f32[8, 197, 1]", rsqrt_16: "f32[8, 197, 1]", view_100: "f32[1576, 256]", getitem_94: "f32[8, 4, 197, 64]", getitem_95: "f32[8, 4, 197, 64]", getitem_96: "f32[8, 4, 197, 64]", getitem_98: "f32[8, 4, 197]", getitem_99: "i32[]", getitem_100: "i32[]", getitem_103: "i64[]", getitem_104: "i64[]", view_104: "f32[1576, 256]", mul_145: "f32[8, 197, 256]", view_106: "f32[1576, 256]", addmm_32: "f32[1576, 768]", view_108: "f32[1576, 768]", mul_150: "f32[8, 197, 256]", view_110: "f32[1576, 256]", getitem_110: "f32[8, 4, 197, 64]", getitem_111: "f32[8, 4, 197, 64]", getitem_112: "f32[8, 4, 197, 64]", getitem_114: "f32[8, 4, 197]", getitem_115: "i32[]", getitem_116: "i32[]", getitem_119: "i64[]", getitem_120: "i64[]", view_114: "f32[1576, 256]", mul_152: "f32[8, 197, 256]", view_116: "f32[1576, 256]", addmm_36: "f32[1576, 768]", view_118: "f32[1576, 768]", mul_157: "f32[8, 197, 256]", view_120: "f32[1576, 256]", getitem_126: "f32[8, 4, 197, 64]", getitem_127: "f32[8, 4, 197, 64]", getitem_128: "f32[8, 4, 197, 64]", getitem_130: "f32[8, 4, 197]", getitem_131: "i32[]", getitem_132: "i32[]", getitem_135: "i64[]", getitem_136: "i64[]", view_124: "f32[1576, 256]", mul_159: "f32[8, 197, 256]", view_126: "f32[1576, 256]", addmm_40: "f32[1576, 768]", view_128: "f32[1576, 768]", mul_164: "f32[8, 1, 128]", view_130: "f32[8, 128]", mul_169: "f32[8, 1, 256]", view_132: "f32[8, 256]", cat_6: "f32[8, 197, 256]", getitem_145: "f32[8, 197, 1]", rsqrt_24: "f32[8, 197, 1]", view_134: "f32[8, 256]", view_137: "f32[1576, 256]", view_150: "f32[8, 256]", mul_177: "f32[8, 1, 256]", view_152: "f32[8, 256]", cat_7: "f32[8, 401, 128]", cat_8: "f32[8, 401, 128]", getitem_149: "f32[8, 401, 1]", rsqrt_26: "f32[8, 401, 1]", view_154: "f32[8, 128]", view_157: "f32[3208, 128]", view_170: "f32[8, 128]", mul_185: "f32[8, 1, 128]", view_172: "f32[8, 128]", cat_9: "f32[8, 197, 256]", getitem_153: "f32[8, 401, 1]", rsqrt_28: "f32[8, 401, 1]", view_174: "f32[3208, 128]", getitem_154: "f32[8, 4, 401, 32]", getitem_155: "f32[8, 4, 401, 32]", getitem_156: "f32[8, 4, 401, 32]", getitem_158: "f32[8, 4, 401]", getitem_159: "i32[]", getitem_160: "i32[]", getitem_163: "i64[]", getitem_164: "i64[]", view_178: "f32[3208, 128]", mul_192: "f32[8, 401, 128]", view_180: "f32[3208, 128]", addmm_54: "f32[3208, 384]", view_182: "f32[3208, 384]", getitem_169: "f32[8, 197, 1]", rsqrt_30: "f32[8, 197, 1]", view_184: "f32[1576, 256]", getitem_170: "f32[8, 4, 197, 64]", getitem_171: "f32[8, 4, 197, 64]", getitem_172: "f32[8, 4, 197, 64]", getitem_174: "f32[8, 4, 197]", getitem_175: "i32[]", getitem_176: "i32[]", getitem_179: "i64[]", getitem_180: "i64[]", view_188: "f32[1576, 256]", mul_199: "f32[8, 197, 256]", view_190: "f32[1576, 256]", addmm_58: "f32[1576, 768]", view_192: "f32[1576, 768]", mul_204: "f32[8, 197, 256]", view_194: "f32[1576, 256]", getitem_186: "f32[8, 4, 197, 64]", getitem_187: "f32[8, 4, 197, 64]", getitem_188: "f32[8, 4, 197, 64]", getitem_190: "f32[8, 4, 197]", getitem_191: "i32[]", getitem_192: "i32[]", getitem_195: "i64[]", getitem_196: "i64[]", view_198: "f32[1576, 256]", mul_206: "f32[8, 197, 256]", view_200: "f32[1576, 256]", addmm_62: "f32[1576, 768]", view_202: "f32[1576, 768]", mul_211: "f32[8, 197, 256]", view_204: "f32[1576, 256]", getitem_202: "f32[8, 4, 197, 64]", getitem_203: "f32[8, 4, 197, 64]", getitem_204: "f32[8, 4, 197, 64]", getitem_206: "f32[8, 4, 197]", getitem_207: "i32[]", getitem_208: "i32[]", getitem_211: "i64[]", getitem_212: "i64[]", view_208: "f32[1576, 256]", mul_213: "f32[8, 197, 256]", view_210: "f32[1576, 256]", addmm_66: "f32[1576, 768]", view_212: "f32[1576, 768]", mul_218: "f32[8, 1, 128]", view_214: "f32[8, 128]", mul_223: "f32[8, 1, 256]", view_216: "f32[8, 256]", cat_10: "f32[8, 197, 256]", getitem_221: "f32[8, 197, 1]", rsqrt_38: "f32[8, 197, 1]", view_218: "f32[8, 256]", view_221: "f32[1576, 256]", view_234: "f32[8, 256]", mul_231: "f32[8, 1, 256]", view_236: "f32[8, 256]", cat_11: "f32[8, 401, 128]", cat_12: "f32[8, 401, 128]", getitem_225: "f32[8, 401, 1]", rsqrt_40: "f32[8, 401, 1]", view_238: "f32[8, 128]", view_241: "f32[3208, 128]", view_254: "f32[8, 128]", mul_239: "f32[8, 1, 128]", view_256: "f32[8, 128]", cat_13: "f32[8, 197, 256]", getitem_229: "f32[8, 401, 1]", rsqrt_42: "f32[8, 401, 1]", getitem_231: "f32[8, 197, 1]", rsqrt_43: "f32[8, 197, 1]", clone_68: "f32[8, 128]", clone_69: "f32[8, 256]", permute_142: "f32[1000, 256]", permute_146: "f32[1000, 128]", permute_150: "f32[256, 128]", div_9: "f32[8, 1, 1]", permute_154: "f32[128, 128]", permute_159: "f32[32, 401, 1]", permute_160: "f32[32, 32, 401]", alias_18: "f32[8, 4, 1, 401]", permute_161: "f32[32, 32, 1]", permute_162: "f32[32, 401, 32]", permute_165: "f32[128, 128]", permute_170: "f32[128, 128]", permute_177: "f32[128, 128]", permute_179: "f32[128, 256]", div_11: "f32[8, 1, 1]", permute_183: "f32[256, 256]", permute_188: "f32[32, 197, 1]", permute_189: "f32[32, 64, 197]", alias_19: "f32[8, 4, 1, 197]", permute_190: "f32[32, 64, 1]", permute_191: "f32[32, 197, 64]", permute_194: "f32[256, 256]", permute_199: "f32[256, 256]", permute_206: "f32[256, 256]", permute_208: "f32[128, 256]", div_13: "f32[8, 1, 1]", permute_212: "f32[256, 128]", div_14: "f32[8, 1, 1]", permute_216: "f32[256, 768]", permute_220: "f32[768, 256]", div_15: "f32[8, 197, 1]", permute_224: "f32[256, 256]", alias_20: "f32[8, 4, 197, 64]", permute_230: "f32[768, 256]", div_16: "f32[8, 197, 1]", permute_234: "f32[256, 768]", permute_238: "f32[768, 256]", div_17: "f32[8, 197, 1]", permute_242: "f32[256, 256]", alias_21: "f32[8, 4, 197, 64]", permute_248: "f32[768, 256]", div_18: "f32[8, 197, 1]", permute_252: "f32[256, 768]", permute_256: "f32[768, 256]", div_19: "f32[8, 197, 1]", permute_260: "f32[256, 256]", alias_22: "f32[8, 4, 197, 64]", permute_266: "f32[768, 256]", permute_270: "f32[128, 384]", permute_274: "f32[384, 128]", div_21: "f32[8, 401, 1]", permute_278: "f32[128, 128]", alias_23: "f32[8, 4, 401, 32]", permute_284: "f32[384, 128]", permute_288: "f32[256, 128]", div_23: "f32[8, 1, 1]", permute_292: "f32[128, 128]", permute_297: "f32[32, 401, 1]", permute_298: "f32[32, 32, 401]", alias_24: "f32[8, 4, 1, 401]", permute_299: "f32[32, 32, 1]", permute_300: "f32[32, 401, 32]", permute_303: "f32[128, 128]", permute_308: "f32[128, 128]", permute_315: "f32[128, 128]", permute_317: "f32[128, 256]", div_25: "f32[8, 1, 1]", permute_321: "f32[256, 256]", permute_326: "f32[32, 197, 1]", permute_327: "f32[32, 64, 197]", alias_25: "f32[8, 4, 1, 197]", permute_328: "f32[32, 64, 1]", permute_329: "f32[32, 197, 64]", permute_332: "f32[256, 256]", permute_337: "f32[256, 256]", permute_344: "f32[256, 256]", permute_346: "f32[128, 256]", div_27: "f32[8, 1, 1]", permute_350: "f32[256, 128]", div_28: "f32[8, 1, 1]", permute_354: "f32[256, 768]", permute_358: "f32[768, 256]", div_29: "f32[8, 197, 1]", permute_362: "f32[256, 256]", alias_26: "f32[8, 4, 197, 64]", permute_368: "f32[768, 256]", div_30: "f32[8, 197, 1]", permute_372: "f32[256, 768]", permute_376: "f32[768, 256]", div_31: "f32[8, 197, 1]", permute_380: "f32[256, 256]", alias_27: "f32[8, 4, 197, 64]", permute_386: "f32[768, 256]", div_32: "f32[8, 197, 1]", permute_390: "f32[256, 768]", permute_394: "f32[768, 256]", div_33: "f32[8, 197, 1]", permute_398: "f32[256, 256]", alias_28: "f32[8, 4, 197, 64]", permute_404: "f32[768, 256]", permute_408: "f32[128, 384]", permute_412: "f32[384, 128]", div_35: "f32[8, 401, 1]", permute_416: "f32[128, 128]", alias_29: "f32[8, 4, 401, 32]", permute_422: "f32[384, 128]", permute_426: "f32[256, 128]", div_37: "f32[8, 1, 1]", permute_430: "f32[128, 128]", permute_435: "f32[32, 401, 1]", permute_436: "f32[32, 32, 401]", alias_30: "f32[8, 4, 1, 401]", permute_437: "f32[32, 32, 1]", permute_438: "f32[32, 401, 32]", permute_441: "f32[128, 128]", permute_446: "f32[128, 128]", permute_453: "f32[128, 128]", permute_455: "f32[128, 256]", div_39: "f32[8, 1, 1]", permute_459: "f32[256, 256]", permute_464: "f32[32, 197, 1]", permute_465: "f32[32, 64, 197]", alias_31: "f32[8, 4, 1, 197]", permute_466: "f32[32, 64, 1]", permute_467: "f32[32, 197, 64]", permute_470: "f32[256, 256]", permute_475: "f32[256, 256]", permute_482: "f32[256, 256]", permute_484: "f32[128, 256]", div_41: "f32[8, 1, 1]", permute_488: "f32[256, 128]", div_42: "f32[8, 1, 1]", permute_492: "f32[256, 768]", permute_496: "f32[768, 256]", div_43: "f32[8, 197, 1]", permute_500: "f32[256, 256]", alias_32: "f32[8, 4, 197, 64]", permute_506: "f32[768, 256]", div_44: "f32[8, 197, 1]", permute_510: "f32[256, 768]", permute_514: "f32[768, 256]", div_45: "f32[8, 197, 1]", permute_518: "f32[256, 256]", alias_33: "f32[8, 4, 197, 64]", permute_524: "f32[768, 256]", div_46: "f32[8, 197, 1]", permute_528: "f32[256, 768]", permute_532: "f32[768, 256]", div_47: "f32[8, 197, 1]", permute_536: "f32[256, 256]", alias_34: "f32[8, 4, 197, 64]", permute_542: "f32[768, 256]", div_48: "f32[8, 197, 1]", permute_546: "f32[128, 384]", permute_550: "f32[384, 128]", div_49: "f32[8, 401, 1]", permute_554: "f32[128, 128]", alias_35: "f32[8, 4, 401, 32]", permute_560: "f32[384, 128]", div_50: "f32[8, 401, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_13: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_2, [8, 401, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_53: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_23: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_6, [8, 197, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476)
    erf_1: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_10, [8, 197, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_2: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_43: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_14, [8, 197, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_3: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    mul_111: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_110, primals_57)
    add_77: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_111, primals_58);  mul_111 = primals_58 = None
    mul_113: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.7071067811865476)
    erf_4: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_78: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_116: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_115, primals_61)
    add_80: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_116, primals_62);  mul_116 = primals_62 = None
    mul_118: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.7071067811865476)
    erf_5: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_81: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_56: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_2, getitem_69);  cat_2 = getitem_69 = None
    mul_120: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_10);  sub_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_124: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_123, primals_75)
    add_87: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_124, primals_76);  mul_124 = primals_76 = None
    mul_126: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.7071067811865476)
    erf_6: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
    add_88: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_59: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_4, getitem_73);  cat_4 = getitem_73 = None
    mul_128: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_12);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_132: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_131, primals_89)
    add_94: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_132, primals_90);  mul_132 = primals_90 = None
    mul_134: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476)
    erf_7: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
    add_95: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_62: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_3, getitem_77);  cat_3 = getitem_77 = None
    mul_136: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_28, [8, 401, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_141: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476)
    erf_8: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_101: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_64: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_5, getitem_93);  cat_5 = getitem_93 = None
    mul_143: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_16);  sub_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_107: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_9: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_36, [8, 197, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_10: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_115: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_40, [8, 197, 768]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476)
    erf_11: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    mul_165: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_164, primals_141)
    add_125: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_165, primals_142);  mul_165 = primals_142 = None
    mul_167: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.7071067811865476)
    erf_12: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_126: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_170: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_169, primals_145)
    add_128: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_170, primals_146);  mul_170 = primals_146 = None
    mul_172: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.7071067811865476)
    erf_13: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_129: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_72: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_6, getitem_145);  cat_6 = getitem_145 = None
    mul_174: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_24);  sub_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_178: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_177, primals_159)
    add_135: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_178, primals_160);  mul_178 = primals_160 = None
    mul_180: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.7071067811865476)
    erf_14: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
    add_136: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_75: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_149);  cat_8 = getitem_149 = None
    mul_182: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_26);  sub_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_186: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_185, primals_173)
    add_142: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_186, primals_174);  mul_186 = primals_174 = None
    mul_188: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476)
    erf_15: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_143: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_78: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_7, getitem_153);  cat_7 = getitem_153 = None
    mul_190: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_28);  sub_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_181: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_54, [8, 401, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_195: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_16: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_149: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_80: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_9, getitem_169);  cat_9 = getitem_169 = None
    mul_197: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_30);  sub_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_191: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_58, [8, 197, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_202: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_17: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_156: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_201: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_62, [8, 197, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476)
    erf_18: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_163: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_66, [8, 197, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476)
    erf_19: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_170: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    mul_219: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_218, primals_225)
    add_173: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_219, primals_226);  mul_219 = primals_226 = None
    mul_221: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.7071067811865476)
    erf_20: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_174: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_224: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_223, primals_229)
    add_176: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_224, primals_230);  mul_224 = primals_230 = None
    mul_226: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.7071067811865476)
    erf_21: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_226);  mul_226 = None
    add_177: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_88: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_10, getitem_221);  cat_10 = getitem_221 = None
    mul_228: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_38);  sub_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_232: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_231, primals_243)
    add_183: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_232, primals_244);  mul_232 = primals_244 = None
    mul_234: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.7071067811865476)
    erf_22: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_184: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_91: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_12, getitem_225);  cat_12 = getitem_225 = None
    mul_236: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_40);  sub_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    mul_240: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_239, primals_257)
    add_190: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_240, primals_258);  mul_240 = primals_258 = None
    mul_242: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.7071067811865476)
    erf_23: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_191: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    sub_94: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_11, getitem_229);  cat_11 = getitem_229 = None
    mul_244: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_42);  sub_94 = None
    sub_95: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_13, getitem_231);  cat_13 = getitem_231 = None
    mul_246: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_43);  sub_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    unsqueeze: "f32[1, 8, 1000]" = torch.ops.aten.unsqueeze.default(tangents_1, 0);  tangents_1 = None
    expand_26: "f32[2, 8, 1000]" = torch.ops.aten.expand.default(unsqueeze, [2, 8, 1000]);  unsqueeze = None
    div_6: "f32[2, 8, 1000]" = torch.ops.aten.div.Scalar(expand_26, 2);  expand_26 = None
    select_2: "f32[8, 1000]" = torch.ops.aten.select.int(div_6, 0, 0)
    select_3: "f32[8, 1000]" = torch.ops.aten.select.int(div_6, 0, 1);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    mm_6: "f32[8, 256]" = torch.ops.aten.mm.default(select_3, permute_142);  permute_142 = None
    permute_143: "f32[1000, 8]" = torch.ops.aten.permute.default(select_3, [1, 0])
    mm_7: "f32[1000, 256]" = torch.ops.aten.mm.default(permute_143, clone_69);  permute_143 = clone_69 = None
    permute_144: "f32[256, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_7: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(select_3, [0], True);  select_3 = None
    view_259: "f32[1000]" = torch.ops.aten.view.default(sum_7, [1000]);  sum_7 = None
    permute_145: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    mm_8: "f32[8, 128]" = torch.ops.aten.mm.default(select_2, permute_146);  permute_146 = None
    permute_147: "f32[1000, 8]" = torch.ops.aten.permute.default(select_2, [1, 0])
    mm_9: "f32[1000, 128]" = torch.ops.aten.mm.default(permute_147, clone_68);  permute_147 = clone_68 = None
    permute_148: "f32[128, 1000]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_8: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(select_2, [0], True);  select_2 = None
    view_260: "f32[1000]" = torch.ops.aten.view.default(sum_8, [1000]);  sum_8 = None
    permute_149: "f32[1000, 128]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    full_default: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 197, 256]" = torch.ops.aten.select_scatter.default(full_default, mm_6, 1, 0);  mm_6 = None
    slice_scatter: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, select_scatter, 0, 0, 9223372036854775807);  select_scatter = None
    full_default_2: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_1: "f32[8, 401, 128]" = torch.ops.aten.select_scatter.default(full_default_2, mm_8, 1, 0);  mm_8 = None
    slice_scatter_1: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, select_scatter_1, 0, 0, 9223372036854775807);  select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    mul_249: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_263);  primals_263 = None
    mul_250: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_249, 256)
    sum_9: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True)
    mul_251: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_249, mul_246);  mul_249 = None
    sum_10: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    mul_252: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_246, sum_10);  sum_10 = None
    sub_97: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_250, sum_9);  mul_250 = sum_9 = None
    sub_98: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_97, mul_252);  sub_97 = mul_252 = None
    div_7: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 256);  rsqrt_43 = None
    mul_253: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_7, sub_98);  div_7 = sub_98 = None
    mul_254: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_246);  mul_246 = None
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1]);  mul_254 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    mul_256: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(slice_scatter_1, primals_261);  primals_261 = None
    mul_257: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_256, 128)
    sum_13: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True)
    mul_258: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_256, mul_244);  mul_256 = None
    sum_14: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True);  mul_258 = None
    mul_259: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_244, sum_14);  sum_14 = None
    sub_100: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_257, sum_13);  mul_257 = sum_13 = None
    sub_101: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_100, mul_259);  sub_100 = mul_259 = None
    div_8: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 128);  rsqrt_42 = None
    mul_260: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_8, sub_101);  div_8 = sub_101 = None
    mul_261: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(slice_scatter_1, mul_244);  mul_244 = None
    sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1]);  mul_261 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(slice_scatter_1, [0, 1]);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_69: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(mul_253, 1, 0, 1)
    slice_70: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(mul_253, 1, 1, 197);  mul_253 = None
    slice_scatter_2: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_70, 1, 1, 9223372036854775807);  slice_70 = None
    slice_scatter_3: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_2, 0, 0, 9223372036854775807);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_261: "f32[8, 256]" = torch.ops.aten.view.default(slice_69, [8, 256]);  slice_69 = None
    mm_10: "f32[8, 128]" = torch.ops.aten.mm.default(view_261, permute_150);  permute_150 = None
    permute_151: "f32[256, 8]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_11: "f32[256, 128]" = torch.ops.aten.mm.default(permute_151, view_256);  permute_151 = view_256 = None
    permute_152: "f32[128, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_17: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[256]" = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
    permute_153: "f32[256, 128]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_263: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_10, [8, 1, 128]);  mm_10 = None
    mul_263: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_191, 0.5);  add_191 = None
    mul_264: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, add_190)
    mul_265: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_264, -0.5);  mul_264 = None
    exp_6: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_265);  mul_265 = None
    mul_266: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_267: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, mul_266);  add_190 = mul_266 = None
    add_197: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_263, mul_267);  mul_263 = mul_267 = None
    mul_268: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_263, add_197);  view_263 = add_197 = None
    mul_270: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_268, primals_257);  primals_257 = None
    mul_271: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_270, 128)
    sum_18: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_270, mul_239);  mul_270 = None
    sum_19: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_239, sum_19);  sum_19 = None
    sub_103: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_271, sum_18);  mul_271 = sum_18 = None
    sub_104: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_103, mul_273);  sub_103 = mul_273 = None
    mul_274: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_9, sub_104);  div_9 = sub_104 = None
    mul_275: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_268, mul_239);  mul_239 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    full_default_6: "f32[8, 1, 128]" = torch.ops.aten.full.default([8, 1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, mul_274, 0, 0, 9223372036854775807);  mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_264: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_4, [8, 128])
    mm_12: "f32[8, 128]" = torch.ops.aten.mm.default(view_264, permute_154);  permute_154 = None
    permute_155: "f32[128, 8]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_13: "f32[128, 128]" = torch.ops.aten.mm.default(permute_155, view_254);  permute_155 = view_254 = None
    permute_156: "f32[128, 128]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[128]" = torch.ops.aten.view.default(sum_22, [128]);  sum_22 = None
    permute_157: "f32[128, 128]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_266: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_12, [8, 1, 128]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_267: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_266, [8, 1, 4, 32]);  view_266 = None
    permute_158: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    view_268: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_158, [32, 1, 32]);  permute_158 = None
    bmm_12: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_159, view_268);  permute_159 = None
    bmm_13: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_268, permute_160);  view_268 = permute_160 = None
    view_269: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_12, [8, 4, 401, 32]);  bmm_12 = None
    view_270: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_13, [8, 4, 1, 401]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_276: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_270, alias_18);  view_270 = None
    sum_23: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
    mul_277: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_18, sum_23);  alias_18 = sum_23 = None
    sub_105: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_278: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_105, 0.1767766952966369);  sub_105 = None
    view_271: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_278, [32, 1, 401]);  mul_278 = None
    bmm_14: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_161, view_271);  permute_161 = None
    bmm_15: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_271, permute_162);  view_271 = permute_162 = None
    view_272: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_14, [8, 4, 32, 401]);  bmm_14 = None
    view_273: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_15, [8, 4, 1, 32]);  bmm_15 = None
    permute_163: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 2]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_164: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_70: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_274: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_70, [8, 401, 128]);  clone_70 = None
    view_275: "f32[3208, 128]" = torch.ops.aten.view.default(view_274, [3208, 128]);  view_274 = None
    mm_14: "f32[3208, 128]" = torch.ops.aten.mm.default(view_275, permute_165);  permute_165 = None
    permute_166: "f32[128, 3208]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_15: "f32[128, 128]" = torch.ops.aten.mm.default(permute_166, view_241);  permute_166 = None
    permute_167: "f32[128, 128]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_24: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[128]" = torch.ops.aten.view.default(sum_24, [128]);  sum_24 = None
    permute_168: "f32[128, 128]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    view_277: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_14, [8, 401, 128]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_169: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_163, [0, 2, 1, 3]);  permute_163 = None
    view_278: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_169, [8, 401, 128]);  permute_169 = None
    clone_71: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_278, memory_format = torch.contiguous_format);  view_278 = None
    view_279: "f32[3208, 128]" = torch.ops.aten.view.default(clone_71, [3208, 128]);  clone_71 = None
    mm_16: "f32[3208, 128]" = torch.ops.aten.mm.default(view_279, permute_170);  permute_170 = None
    permute_171: "f32[128, 3208]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_17: "f32[128, 128]" = torch.ops.aten.mm.default(permute_171, view_241);  permute_171 = view_241 = None
    permute_172: "f32[128, 128]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[128]" = torch.ops.aten.view.default(sum_25, [128]);  sum_25 = None
    permute_173: "f32[128, 128]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_281: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_16, [8, 401, 128]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_198: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_277, view_281);  view_277 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_174: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    view_282: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_174, [8, 1, 128]);  permute_174 = None
    sum_26: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 1], True)
    view_283: "f32[128]" = torch.ops.aten.view.default(sum_26, [128]);  sum_26 = None
    view_284: "f32[8, 128]" = torch.ops.aten.view.default(view_282, [8, 128]);  view_282 = None
    permute_175: "f32[128, 8]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_18: "f32[128, 128]" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
    permute_176: "f32[128, 128]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    mm_19: "f32[8, 128]" = torch.ops.aten.mm.default(view_284, permute_177);  view_284 = permute_177 = None
    view_285: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_19, [8, 1, 128]);  mm_19 = None
    permute_178: "f32[128, 128]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    slice_scatter_5: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, view_285, 1, 0, 1);  view_285 = None
    slice_scatter_6: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_5, 0, 0, 9223372036854775807);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_199: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_198, slice_scatter_6);  add_198 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_280: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_199, primals_247);  primals_247 = None
    mul_281: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_280, 128)
    sum_27: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_280, mul_236);  mul_280 = None
    sum_28: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_236, sum_28);  sum_28 = None
    sub_107: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_281, sum_27);  mul_281 = sum_27 = None
    sub_108: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_107, mul_283);  sub_107 = mul_283 = None
    div_10: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 128);  rsqrt_40 = None
    mul_284: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_10, sub_108);  div_10 = sub_108 = None
    mul_285: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_199, mul_236);  mul_236 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_199, [0, 1]);  add_199 = None
    slice_scatter_7: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_4, 1, 0, 1);  slice_scatter_4 = None
    slice_scatter_8: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_200: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_284, slice_scatter_8);  mul_284 = slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_71: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_200, 1, 0, 1)
    slice_72: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_200, 1, 1, 401);  add_200 = None
    slice_scatter_9: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_72, 1, 1, 9223372036854775807);  slice_72 = None
    slice_scatter_10: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_73: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(mul_260, 1, 0, 1)
    slice_74: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(mul_260, 1, 1, 401);  mul_260 = None
    slice_scatter_11: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_74, 1, 1, 9223372036854775807);  slice_74 = None
    slice_scatter_12: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_11, 0, 0, 9223372036854775807);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_201: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_12);  slice_scatter_10 = slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_286: "f32[8, 128]" = torch.ops.aten.view.default(slice_73, [8, 128]);  slice_73 = None
    mm_20: "f32[8, 256]" = torch.ops.aten.mm.default(view_286, permute_179);  permute_179 = None
    permute_180: "f32[128, 8]" = torch.ops.aten.permute.default(view_286, [1, 0])
    mm_21: "f32[128, 256]" = torch.ops.aten.mm.default(permute_180, view_236);  permute_180 = view_236 = None
    permute_181: "f32[256, 128]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_286, [0], True);  view_286 = None
    view_287: "f32[128]" = torch.ops.aten.view.default(sum_31, [128]);  sum_31 = None
    permute_182: "f32[128, 256]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_288: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_20, [8, 1, 256]);  mm_20 = None
    mul_287: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
    mul_288: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, add_183)
    mul_289: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_288, -0.5);  mul_288 = None
    exp_7: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_289);  mul_289 = None
    mul_290: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_291: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, mul_290);  add_183 = mul_290 = None
    add_203: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_287, mul_291);  mul_287 = mul_291 = None
    mul_292: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_288, add_203);  view_288 = add_203 = None
    mul_294: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_292, primals_243);  primals_243 = None
    mul_295: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_294, 256)
    sum_32: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True)
    mul_296: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_294, mul_231);  mul_294 = None
    sum_33: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True);  mul_296 = None
    mul_297: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_231, sum_33);  sum_33 = None
    sub_110: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_295, sum_32);  mul_295 = sum_32 = None
    sub_111: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_110, mul_297);  sub_110 = mul_297 = None
    mul_298: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_11, sub_111);  div_11 = sub_111 = None
    mul_299: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_292, mul_231);  mul_231 = None
    sum_34: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1]);  mul_299 = None
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
    full_default_15: "f32[8, 1, 256]" = torch.ops.aten.full.default([8, 1, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_default_15, mul_298, 0, 0, 9223372036854775807);  mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_289: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_13, [8, 256])
    mm_22: "f32[8, 256]" = torch.ops.aten.mm.default(view_289, permute_183);  permute_183 = None
    permute_184: "f32[256, 8]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_184, view_234);  permute_184 = view_234 = None
    permute_185: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[256]" = torch.ops.aten.view.default(sum_36, [256]);  sum_36 = None
    permute_186: "f32[256, 256]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_291: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_22, [8, 1, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_292: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_291, [8, 1, 4, 64]);  view_291 = None
    permute_187: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    view_293: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_187, [32, 1, 64]);  permute_187 = None
    bmm_16: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_188, view_293);  permute_188 = None
    bmm_17: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_293, permute_189);  view_293 = permute_189 = None
    view_294: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_16, [8, 4, 197, 64]);  bmm_16 = None
    view_295: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_17, [8, 4, 1, 197]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_300: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_295, alias_19);  view_295 = None
    sum_37: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [-1], True)
    mul_301: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_19, sum_37);  alias_19 = sum_37 = None
    sub_112: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_302: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_112, 0.125);  sub_112 = None
    view_296: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_302, [32, 1, 197]);  mul_302 = None
    bmm_18: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_190, view_296);  permute_190 = None
    bmm_19: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_296, permute_191);  view_296 = permute_191 = None
    view_297: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_18, [8, 4, 64, 197]);  bmm_18 = None
    view_298: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_19, [8, 4, 1, 64]);  bmm_19 = None
    permute_192: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_193: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_72: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_299: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_72, [8, 197, 256]);  clone_72 = None
    view_300: "f32[1576, 256]" = torch.ops.aten.view.default(view_299, [1576, 256]);  view_299 = None
    mm_24: "f32[1576, 256]" = torch.ops.aten.mm.default(view_300, permute_194);  permute_194 = None
    permute_195: "f32[256, 1576]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_195, view_221);  permute_195 = None
    permute_196: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_38: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[256]" = torch.ops.aten.view.default(sum_38, [256]);  sum_38 = None
    permute_197: "f32[256, 256]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_302: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_24, [8, 197, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_198: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
    view_303: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_198, [8, 197, 256]);  permute_198 = None
    clone_73: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_303, memory_format = torch.contiguous_format);  view_303 = None
    view_304: "f32[1576, 256]" = torch.ops.aten.view.default(clone_73, [1576, 256]);  clone_73 = None
    mm_26: "f32[1576, 256]" = torch.ops.aten.mm.default(view_304, permute_199);  permute_199 = None
    permute_200: "f32[256, 1576]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_27: "f32[256, 256]" = torch.ops.aten.mm.default(permute_200, view_221);  permute_200 = view_221 = None
    permute_201: "f32[256, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_39: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[256]" = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
    permute_202: "f32[256, 256]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_306: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_26, [8, 197, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_204: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_302, view_306);  view_302 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_203: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
    view_307: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_203, [8, 1, 256]);  permute_203 = None
    sum_40: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1], True)
    view_308: "f32[256]" = torch.ops.aten.view.default(sum_40, [256]);  sum_40 = None
    view_309: "f32[8, 256]" = torch.ops.aten.view.default(view_307, [8, 256]);  view_307 = None
    permute_204: "f32[256, 8]" = torch.ops.aten.permute.default(view_309, [1, 0])
    mm_28: "f32[256, 256]" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
    permute_205: "f32[256, 256]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    mm_29: "f32[8, 256]" = torch.ops.aten.mm.default(view_309, permute_206);  view_309 = permute_206 = None
    view_310: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_29, [8, 1, 256]);  mm_29 = None
    permute_207: "f32[256, 256]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    slice_scatter_14: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, view_310, 1, 0, 1);  view_310 = None
    slice_scatter_15: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_14, 0, 0, 9223372036854775807);  slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_205: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_204, slice_scatter_15);  add_204 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_304: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_205, primals_233);  primals_233 = None
    mul_305: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_304, 256)
    sum_41: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
    mul_306: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_304, mul_228);  mul_304 = None
    sum_42: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
    mul_307: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_228, sum_42);  sum_42 = None
    sub_114: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_305, sum_41);  mul_305 = sum_41 = None
    sub_115: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_114, mul_307);  sub_114 = mul_307 = None
    div_12: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 256);  rsqrt_38 = None
    mul_308: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_12, sub_115);  div_12 = sub_115 = None
    mul_309: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_205, mul_228);  mul_228 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_205, [0, 1]);  add_205 = None
    slice_scatter_16: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_13, 1, 0, 1);  slice_scatter_13 = None
    slice_scatter_17: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_16, 0, 0, 9223372036854775807);  slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_206: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_308, slice_scatter_17);  mul_308 = slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_75: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_206, 1, 0, 1)
    slice_76: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_206, 1, 1, 197);  add_206 = None
    slice_scatter_18: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_76, 1, 1, 9223372036854775807);  slice_76 = None
    slice_scatter_19: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_18, 0, 0, 9223372036854775807);  slice_scatter_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_207: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_3, slice_scatter_19);  slice_scatter_3 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_311: "f32[8, 128]" = torch.ops.aten.view.default(slice_71, [8, 128]);  slice_71 = None
    mm_30: "f32[8, 256]" = torch.ops.aten.mm.default(view_311, permute_208);  permute_208 = None
    permute_209: "f32[128, 8]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_31: "f32[128, 256]" = torch.ops.aten.mm.default(permute_209, view_216);  permute_209 = view_216 = None
    permute_210: "f32[256, 128]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_45: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[128]" = torch.ops.aten.view.default(sum_45, [128]);  sum_45 = None
    permute_211: "f32[128, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_313: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_30, [8, 1, 256]);  mm_30 = None
    mul_311: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_177, 0.5);  add_177 = None
    mul_312: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, add_176)
    mul_313: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_312, -0.5);  mul_312 = None
    exp_8: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_313);  mul_313 = None
    mul_314: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_315: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, mul_314);  add_176 = mul_314 = None
    add_209: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_311, mul_315);  mul_311 = mul_315 = None
    mul_316: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_313, add_209);  view_313 = add_209 = None
    mul_318: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_316, primals_229);  primals_229 = None
    mul_319: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_318, 256)
    sum_46: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_318, mul_223);  mul_318 = None
    sum_47: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_223, sum_47);  sum_47 = None
    sub_117: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_319, sum_46);  mul_319 = sum_46 = None
    sub_118: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_117, mul_321);  sub_117 = mul_321 = None
    mul_322: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_13, sub_118);  div_13 = sub_118 = None
    mul_323: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_316, mul_223);  mul_223 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    slice_scatter_20: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, mul_322, 1, 0, 1);  mul_322 = None
    slice_scatter_21: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_20, 0, 0, 9223372036854775807);  slice_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_210: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_207, slice_scatter_21);  add_207 = slice_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_314: "f32[8, 256]" = torch.ops.aten.view.default(slice_75, [8, 256]);  slice_75 = None
    mm_32: "f32[8, 128]" = torch.ops.aten.mm.default(view_314, permute_212);  permute_212 = None
    permute_213: "f32[256, 8]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_33: "f32[256, 128]" = torch.ops.aten.mm.default(permute_213, view_214);  permute_213 = view_214 = None
    permute_214: "f32[128, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_215: "f32[256, 128]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_316: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_32, [8, 1, 128]);  mm_32 = None
    mul_325: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_174, 0.5);  add_174 = None
    mul_326: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, add_173)
    mul_327: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_9: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_329: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, mul_328);  add_173 = mul_328 = None
    add_212: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_316, add_212);  view_316 = add_212 = None
    mul_332: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_330, primals_225);  primals_225 = None
    mul_333: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_332, 128)
    sum_51: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_332, mul_218);  mul_332 = None
    sum_52: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_218, sum_52);  sum_52 = None
    sub_120: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_333, sum_51);  mul_333 = sum_51 = None
    sub_121: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_120, mul_335);  sub_120 = mul_335 = None
    mul_336: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_14, sub_121);  div_14 = sub_121 = None
    mul_337: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_330, mul_218);  mul_218 = None
    sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    slice_scatter_22: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_336, 1, 0, 1);  mul_336 = None
    slice_scatter_23: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_22, 0, 0, 9223372036854775807);  slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_213: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_201, slice_scatter_23);  add_201 = slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_317: "f32[1576, 256]" = torch.ops.aten.view.default(add_210, [1576, 256])
    mm_34: "f32[1576, 768]" = torch.ops.aten.mm.default(view_317, permute_216);  permute_216 = None
    permute_217: "f32[256, 1576]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_35: "f32[256, 768]" = torch.ops.aten.mm.default(permute_217, view_212);  permute_217 = view_212 = None
    permute_218: "f32[768, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_55: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_55, [256]);  sum_55 = None
    permute_219: "f32[256, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_319: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_34, [8, 197, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_170, 0.5);  add_170 = None
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, view_211)
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
    exp_10: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_342);  view_211 = mul_342 = None
    add_215: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, add_215);  view_319 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_320: "f32[1576, 768]" = torch.ops.aten.view.default(mul_344, [1576, 768]);  mul_344 = None
    mm_36: "f32[1576, 256]" = torch.ops.aten.mm.default(view_320, permute_220);  permute_220 = None
    permute_221: "f32[768, 1576]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_37: "f32[768, 256]" = torch.ops.aten.mm.default(permute_221, view_210);  permute_221 = view_210 = None
    permute_222: "f32[256, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_223: "f32[768, 256]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_322: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_36, [8, 197, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_346: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_322, primals_219);  primals_219 = None
    mul_347: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_346, 256)
    sum_57: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
    mul_348: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_346, mul_213);  mul_346 = None
    sum_58: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
    mul_349: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_213, sum_58);  sum_58 = None
    sub_123: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_347, sum_57);  mul_347 = sum_57 = None
    sub_124: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_123, mul_349);  sub_123 = mul_349 = None
    mul_350: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_15, sub_124);  div_15 = sub_124 = None
    mul_351: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_322, mul_213);  mul_213 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_322, [0, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_216: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_210, mul_350);  add_210 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_323: "f32[1576, 256]" = torch.ops.aten.view.default(add_216, [1576, 256])
    mm_38: "f32[1576, 256]" = torch.ops.aten.mm.default(view_323, permute_224);  permute_224 = None
    permute_225: "f32[256, 1576]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_39: "f32[256, 256]" = torch.ops.aten.mm.default(permute_225, view_208);  permute_225 = view_208 = None
    permute_226: "f32[256, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_61: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[256]" = torch.ops.aten.view.default(sum_61, [256]);  sum_61 = None
    permute_227: "f32[256, 256]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_325: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_38, [8, 197, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_326: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_325, [8, 197, 4, 64]);  view_325 = None
    permute_228: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_228, getitem_202, getitem_203, getitem_204, alias_20, getitem_206, getitem_207, getitem_208, 0, 0, 0.0, False, getitem_211, getitem_212);  permute_228 = getitem_202 = getitem_203 = getitem_204 = alias_20 = getitem_206 = getitem_207 = getitem_208 = getitem_211 = getitem_212 = None
    getitem_232: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_233: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_234: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_232, getitem_233, getitem_234]);  getitem_232 = getitem_233 = getitem_234 = None
    view_327: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_15, [3, 8, 4, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_229: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_327, [1, 3, 0, 2, 4]);  view_327 = None
    clone_76: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    view_328: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_76, [8, 197, 768]);  clone_76 = None
    view_329: "f32[1576, 768]" = torch.ops.aten.view.default(view_328, [1576, 768]);  view_328 = None
    mm_40: "f32[1576, 256]" = torch.ops.aten.mm.default(view_329, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_41: "f32[768, 256]" = torch.ops.aten.mm.default(permute_231, view_204);  permute_231 = view_204 = None
    permute_232: "f32[256, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_233: "f32[768, 256]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_331: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_40, [8, 197, 256]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_353: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_331, primals_213);  primals_213 = None
    mul_354: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_353, 256)
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_353, mul_211);  mul_353 = None
    sum_64: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_211, sum_64);  sum_64 = None
    sub_126: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_354, sum_63);  mul_354 = sum_63 = None
    sub_127: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_126, mul_356);  sub_126 = mul_356 = None
    mul_357: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_16, sub_127);  div_16 = sub_127 = None
    mul_358: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_331, mul_211);  mul_211 = None
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_331, [0, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_217: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_216, mul_357);  add_216 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_332: "f32[1576, 256]" = torch.ops.aten.view.default(add_217, [1576, 256])
    mm_42: "f32[1576, 768]" = torch.ops.aten.mm.default(view_332, permute_234);  permute_234 = None
    permute_235: "f32[256, 1576]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_43: "f32[256, 768]" = torch.ops.aten.mm.default(permute_235, view_202);  permute_235 = view_202 = None
    permute_236: "f32[768, 256]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_67: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[256]" = torch.ops.aten.view.default(sum_67, [256]);  sum_67 = None
    permute_237: "f32[256, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_334: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_42, [8, 197, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_360: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_163, 0.5);  add_163 = None
    mul_361: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, view_201)
    mul_362: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_361, -0.5);  mul_361 = None
    exp_11: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_362);  mul_362 = None
    mul_363: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_364: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, mul_363);  view_201 = mul_363 = None
    add_219: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_360, mul_364);  mul_360 = mul_364 = None
    mul_365: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_334, add_219);  view_334 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_335: "f32[1576, 768]" = torch.ops.aten.view.default(mul_365, [1576, 768]);  mul_365 = None
    mm_44: "f32[1576, 256]" = torch.ops.aten.mm.default(view_335, permute_238);  permute_238 = None
    permute_239: "f32[768, 1576]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_45: "f32[768, 256]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[256, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    permute_241: "f32[768, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_337: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_44, [8, 197, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_367: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_337, primals_207);  primals_207 = None
    mul_368: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_367, 256)
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_367, mul_206);  mul_367 = None
    sum_70: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_206, sum_70);  sum_70 = None
    sub_129: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_368, sum_69);  mul_368 = sum_69 = None
    sub_130: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_129, mul_370);  sub_129 = mul_370 = None
    mul_371: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_17, sub_130);  div_17 = sub_130 = None
    mul_372: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_337, mul_206);  mul_206 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_72: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_337, [0, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_220: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_217, mul_371);  add_217 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_338: "f32[1576, 256]" = torch.ops.aten.view.default(add_220, [1576, 256])
    mm_46: "f32[1576, 256]" = torch.ops.aten.mm.default(view_338, permute_242);  permute_242 = None
    permute_243: "f32[256, 1576]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_73: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[256]" = torch.ops.aten.view.default(sum_73, [256]);  sum_73 = None
    permute_245: "f32[256, 256]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_340: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_46, [8, 197, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_341: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_340, [8, 197, 4, 64]);  view_340 = None
    permute_246: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_246, getitem_186, getitem_187, getitem_188, alias_21, getitem_190, getitem_191, getitem_192, 0, 0, 0.0, False, getitem_195, getitem_196);  permute_246 = getitem_186 = getitem_187 = getitem_188 = alias_21 = getitem_190 = getitem_191 = getitem_192 = getitem_195 = getitem_196 = None
    getitem_235: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_236: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_237: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_16: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_235, getitem_236, getitem_237]);  getitem_235 = getitem_236 = getitem_237 = None
    view_342: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_16, [3, 8, 4, 197, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_247: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_342, [1, 3, 0, 2, 4]);  view_342 = None
    clone_77: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_343: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_77, [8, 197, 768]);  clone_77 = None
    view_344: "f32[1576, 768]" = torch.ops.aten.view.default(view_343, [1576, 768]);  view_343 = None
    mm_48: "f32[1576, 256]" = torch.ops.aten.mm.default(view_344, permute_248);  permute_248 = None
    permute_249: "f32[768, 1576]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_49: "f32[768, 256]" = torch.ops.aten.mm.default(permute_249, view_194);  permute_249 = view_194 = None
    permute_250: "f32[256, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_251: "f32[768, 256]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_346: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_48, [8, 197, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_374: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_346, primals_201);  primals_201 = None
    mul_375: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_374, 256)
    sum_75: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True)
    mul_376: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_374, mul_204);  mul_374 = None
    sum_76: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
    mul_377: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_204, sum_76);  sum_76 = None
    sub_132: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_375, sum_75);  mul_375 = sum_75 = None
    sub_133: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_132, mul_377);  sub_132 = mul_377 = None
    mul_378: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_18, sub_133);  div_18 = sub_133 = None
    mul_379: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_346, mul_204);  mul_204 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_346, [0, 1]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_221: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_220, mul_378);  add_220 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_347: "f32[1576, 256]" = torch.ops.aten.view.default(add_221, [1576, 256])
    mm_50: "f32[1576, 768]" = torch.ops.aten.mm.default(view_347, permute_252);  permute_252 = None
    permute_253: "f32[256, 1576]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_51: "f32[256, 768]" = torch.ops.aten.mm.default(permute_253, view_192);  permute_253 = view_192 = None
    permute_254: "f32[768, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_79: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[256]" = torch.ops.aten.view.default(sum_79, [256]);  sum_79 = None
    permute_255: "f32[256, 768]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    view_349: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_50, [8, 197, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_381: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_156, 0.5);  add_156 = None
    mul_382: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, view_191)
    mul_383: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
    exp_12: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_383);  mul_383 = None
    mul_384: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_385: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, mul_384);  view_191 = mul_384 = None
    add_223: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
    mul_386: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_349, add_223);  view_349 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_350: "f32[1576, 768]" = torch.ops.aten.view.default(mul_386, [1576, 768]);  mul_386 = None
    mm_52: "f32[1576, 256]" = torch.ops.aten.mm.default(view_350, permute_256);  permute_256 = None
    permute_257: "f32[768, 1576]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_53: "f32[768, 256]" = torch.ops.aten.mm.default(permute_257, view_190);  permute_257 = view_190 = None
    permute_258: "f32[256, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_259: "f32[768, 256]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    view_352: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_52, [8, 197, 256]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_388: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_352, primals_195);  primals_195 = None
    mul_389: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_388, 256)
    sum_81: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_388, mul_199);  mul_388 = None
    sum_82: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_199, sum_82);  sum_82 = None
    sub_135: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_389, sum_81);  mul_389 = sum_81 = None
    sub_136: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_135, mul_391);  sub_135 = mul_391 = None
    mul_392: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_19, sub_136);  div_19 = sub_136 = None
    mul_393: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_352, mul_199);  mul_199 = None
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_352, [0, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_224: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_221, mul_392);  add_221 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_353: "f32[1576, 256]" = torch.ops.aten.view.default(add_224, [1576, 256])
    mm_54: "f32[1576, 256]" = torch.ops.aten.mm.default(view_353, permute_260);  permute_260 = None
    permute_261: "f32[256, 1576]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_55: "f32[256, 256]" = torch.ops.aten.mm.default(permute_261, view_188);  permute_261 = view_188 = None
    permute_262: "f32[256, 256]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_85: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.view.default(sum_85, [256]);  sum_85 = None
    permute_263: "f32[256, 256]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_355: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_54, [8, 197, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_356: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_355, [8, 197, 4, 64]);  view_355 = None
    permute_264: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_264, getitem_170, getitem_171, getitem_172, alias_22, getitem_174, getitem_175, getitem_176, 0, 0, 0.0, False, getitem_179, getitem_180);  permute_264 = getitem_170 = getitem_171 = getitem_172 = alias_22 = getitem_174 = getitem_175 = getitem_176 = getitem_179 = getitem_180 = None
    getitem_238: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[0]
    getitem_239: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[1]
    getitem_240: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_17: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_238, getitem_239, getitem_240]);  getitem_238 = getitem_239 = getitem_240 = None
    view_357: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_17, [3, 8, 4, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_265: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_357, [1, 3, 0, 2, 4]);  view_357 = None
    clone_78: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_358: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_78, [8, 197, 768]);  clone_78 = None
    view_359: "f32[1576, 768]" = torch.ops.aten.view.default(view_358, [1576, 768]);  view_358 = None
    mm_56: "f32[1576, 256]" = torch.ops.aten.mm.default(view_359, permute_266);  permute_266 = None
    permute_267: "f32[768, 1576]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_57: "f32[768, 256]" = torch.ops.aten.mm.default(permute_267, view_184);  permute_267 = view_184 = None
    permute_268: "f32[256, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    permute_269: "f32[768, 256]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_361: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_56, [8, 197, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_395: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_361, primals_189);  primals_189 = None
    mul_396: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_395, 256)
    sum_87: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_395, mul_197);  mul_395 = None
    sum_88: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_197, sum_88);  sum_88 = None
    sub_138: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_396, sum_87);  mul_396 = sum_87 = None
    sub_139: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_138, mul_398);  sub_138 = mul_398 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 256);  rsqrt_30 = None
    mul_399: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_20, sub_139);  div_20 = sub_139 = None
    mul_400: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_361, mul_197);  mul_197 = None
    sum_89: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_90: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_225: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_224, mul_399);  add_224 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_362: "f32[3208, 128]" = torch.ops.aten.view.default(add_213, [3208, 128])
    mm_58: "f32[3208, 384]" = torch.ops.aten.mm.default(view_362, permute_270);  permute_270 = None
    permute_271: "f32[128, 3208]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_59: "f32[128, 384]" = torch.ops.aten.mm.default(permute_271, view_182);  permute_271 = view_182 = None
    permute_272: "f32[384, 128]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_91: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[128]" = torch.ops.aten.view.default(sum_91, [128]);  sum_91 = None
    permute_273: "f32[128, 384]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_364: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_58, [8, 401, 384]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_402: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_149, 0.5);  add_149 = None
    mul_403: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, view_181)
    mul_404: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_403, -0.5);  mul_403 = None
    exp_13: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_404);  mul_404 = None
    mul_405: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_406: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, mul_405);  view_181 = mul_405 = None
    add_227: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_402, mul_406);  mul_402 = mul_406 = None
    mul_407: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_364, add_227);  view_364 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_365: "f32[3208, 384]" = torch.ops.aten.view.default(mul_407, [3208, 384]);  mul_407 = None
    mm_60: "f32[3208, 128]" = torch.ops.aten.mm.default(view_365, permute_274);  permute_274 = None
    permute_275: "f32[384, 3208]" = torch.ops.aten.permute.default(view_365, [1, 0])
    mm_61: "f32[384, 128]" = torch.ops.aten.mm.default(permute_275, view_180);  permute_275 = view_180 = None
    permute_276: "f32[128, 384]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
    view_366: "f32[384]" = torch.ops.aten.view.default(sum_92, [384]);  sum_92 = None
    permute_277: "f32[384, 128]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_367: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_60, [8, 401, 128]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_409: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_367, primals_183);  primals_183 = None
    mul_410: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_409, 128)
    sum_93: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
    mul_411: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_409, mul_192);  mul_409 = None
    sum_94: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
    mul_412: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_192, sum_94);  sum_94 = None
    sub_141: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_410, sum_93);  mul_410 = sum_93 = None
    sub_142: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_141, mul_412);  sub_141 = mul_412 = None
    mul_413: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_21, sub_142);  div_21 = sub_142 = None
    mul_414: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_367, mul_192);  mul_192 = None
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
    sum_96: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_367, [0, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_228: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_213, mul_413);  add_213 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_368: "f32[3208, 128]" = torch.ops.aten.view.default(add_228, [3208, 128])
    mm_62: "f32[3208, 128]" = torch.ops.aten.mm.default(view_368, permute_278);  permute_278 = None
    permute_279: "f32[128, 3208]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_63: "f32[128, 128]" = torch.ops.aten.mm.default(permute_279, view_178);  permute_279 = view_178 = None
    permute_280: "f32[128, 128]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[128]" = torch.ops.aten.view.default(sum_97, [128]);  sum_97 = None
    permute_281: "f32[128, 128]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_370: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_62, [8, 401, 128]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_371: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_370, [8, 401, 4, 32]);  view_370 = None
    permute_282: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_282, getitem_154, getitem_155, getitem_156, alias_23, getitem_158, getitem_159, getitem_160, 0, 0, 0.0, False, getitem_163, getitem_164);  permute_282 = getitem_154 = getitem_155 = getitem_156 = alias_23 = getitem_158 = getitem_159 = getitem_160 = getitem_163 = getitem_164 = None
    getitem_241: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[0]
    getitem_242: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[1]
    getitem_243: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_18: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_241, getitem_242, getitem_243]);  getitem_241 = getitem_242 = getitem_243 = None
    view_372: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 4, 401, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_283: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_372, [1, 3, 0, 2, 4]);  view_372 = None
    clone_79: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    view_373: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_79, [8, 401, 384]);  clone_79 = None
    view_374: "f32[3208, 384]" = torch.ops.aten.view.default(view_373, [3208, 384]);  view_373 = None
    mm_64: "f32[3208, 128]" = torch.ops.aten.mm.default(view_374, permute_284);  permute_284 = None
    permute_285: "f32[384, 3208]" = torch.ops.aten.permute.default(view_374, [1, 0])
    mm_65: "f32[384, 128]" = torch.ops.aten.mm.default(permute_285, view_174);  permute_285 = view_174 = None
    permute_286: "f32[128, 384]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
    view_375: "f32[384]" = torch.ops.aten.view.default(sum_98, [384]);  sum_98 = None
    permute_287: "f32[384, 128]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_376: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_64, [8, 401, 128]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_416: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_376, primals_177);  primals_177 = None
    mul_417: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_416, 128)
    sum_99: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_416, mul_190);  mul_416 = None
    sum_100: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_190, sum_100);  sum_100 = None
    sub_144: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_417, sum_99);  mul_417 = sum_99 = None
    sub_145: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_144, mul_419);  sub_144 = mul_419 = None
    div_22: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 128);  rsqrt_28 = None
    mul_420: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_22, sub_145);  div_22 = sub_145 = None
    mul_421: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_376, mul_190);  mul_190 = None
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_102: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_376, [0, 1]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_229: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_228, mul_420);  add_228 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_77: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_225, 1, 0, 1)
    slice_78: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_225, 1, 1, 197);  add_225 = None
    slice_scatter_24: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_78, 1, 1, 9223372036854775807);  slice_78 = None
    slice_scatter_25: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_24, 0, 0, 9223372036854775807);  slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_377: "f32[8, 256]" = torch.ops.aten.view.default(slice_77, [8, 256]);  slice_77 = None
    mm_66: "f32[8, 128]" = torch.ops.aten.mm.default(view_377, permute_288);  permute_288 = None
    permute_289: "f32[256, 8]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_67: "f32[256, 128]" = torch.ops.aten.mm.default(permute_289, view_172);  permute_289 = view_172 = None
    permute_290: "f32[128, 256]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[256]" = torch.ops.aten.view.default(sum_103, [256]);  sum_103 = None
    permute_291: "f32[256, 128]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_379: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_66, [8, 1, 128]);  mm_66 = None
    mul_423: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_143, 0.5);  add_143 = None
    mul_424: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, add_142)
    mul_425: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_424, -0.5);  mul_424 = None
    exp_14: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_425);  mul_425 = None
    mul_426: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_427: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, mul_426);  add_142 = mul_426 = None
    add_231: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_423, mul_427);  mul_423 = mul_427 = None
    mul_428: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_379, add_231);  view_379 = add_231 = None
    mul_430: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_428, primals_173);  primals_173 = None
    mul_431: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_430, 128)
    sum_104: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True)
    mul_432: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_430, mul_185);  mul_430 = None
    sum_105: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
    mul_433: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_185, sum_105);  sum_105 = None
    sub_147: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_431, sum_104);  mul_431 = sum_104 = None
    sub_148: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_147, mul_433);  sub_147 = mul_433 = None
    mul_434: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_23, sub_148);  div_23 = sub_148 = None
    mul_435: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_428, mul_185);  mul_185 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
    sum_107: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
    slice_scatter_26: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, mul_434, 0, 0, 9223372036854775807);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_380: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_26, [8, 128])
    mm_68: "f32[8, 128]" = torch.ops.aten.mm.default(view_380, permute_292);  permute_292 = None
    permute_293: "f32[128, 8]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_69: "f32[128, 128]" = torch.ops.aten.mm.default(permute_293, view_170);  permute_293 = view_170 = None
    permute_294: "f32[128, 128]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[128]" = torch.ops.aten.view.default(sum_108, [128]);  sum_108 = None
    permute_295: "f32[128, 128]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    view_382: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_68, [8, 1, 128]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_383: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_382, [8, 1, 4, 32]);  view_382 = None
    permute_296: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    view_384: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_296, [32, 1, 32]);  permute_296 = None
    bmm_20: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_297, view_384);  permute_297 = None
    bmm_21: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_384, permute_298);  view_384 = permute_298 = None
    view_385: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_20, [8, 4, 401, 32]);  bmm_20 = None
    view_386: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_21, [8, 4, 1, 401]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_436: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_386, alias_24);  view_386 = None
    sum_109: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [-1], True)
    mul_437: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_24, sum_109);  alias_24 = sum_109 = None
    sub_149: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_438: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_149, 0.1767766952966369);  sub_149 = None
    view_387: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_438, [32, 1, 401]);  mul_438 = None
    bmm_22: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_299, view_387);  permute_299 = None
    bmm_23: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_387, permute_300);  view_387 = permute_300 = None
    view_388: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_22, [8, 4, 32, 401]);  bmm_22 = None
    view_389: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_23, [8, 4, 1, 32]);  bmm_23 = None
    permute_301: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_388, [0, 1, 3, 2]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_302: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None
    clone_80: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
    view_390: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_80, [8, 401, 128]);  clone_80 = None
    view_391: "f32[3208, 128]" = torch.ops.aten.view.default(view_390, [3208, 128]);  view_390 = None
    mm_70: "f32[3208, 128]" = torch.ops.aten.mm.default(view_391, permute_303);  permute_303 = None
    permute_304: "f32[128, 3208]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_71: "f32[128, 128]" = torch.ops.aten.mm.default(permute_304, view_157);  permute_304 = None
    permute_305: "f32[128, 128]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[128]" = torch.ops.aten.view.default(sum_110, [128]);  sum_110 = None
    permute_306: "f32[128, 128]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_393: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_70, [8, 401, 128]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_307: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
    view_394: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_307, [8, 401, 128]);  permute_307 = None
    clone_81: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_394, memory_format = torch.contiguous_format);  view_394 = None
    view_395: "f32[3208, 128]" = torch.ops.aten.view.default(clone_81, [3208, 128]);  clone_81 = None
    mm_72: "f32[3208, 128]" = torch.ops.aten.mm.default(view_395, permute_308);  permute_308 = None
    permute_309: "f32[128, 3208]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_73: "f32[128, 128]" = torch.ops.aten.mm.default(permute_309, view_157);  permute_309 = view_157 = None
    permute_310: "f32[128, 128]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[128]" = torch.ops.aten.view.default(sum_111, [128]);  sum_111 = None
    permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_397: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_72, [8, 401, 128]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_232: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_393, view_397);  view_393 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_312: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
    view_398: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_312, [8, 1, 128]);  permute_312 = None
    sum_112: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_398, [0, 1], True)
    view_399: "f32[128]" = torch.ops.aten.view.default(sum_112, [128]);  sum_112 = None
    view_400: "f32[8, 128]" = torch.ops.aten.view.default(view_398, [8, 128]);  view_398 = None
    permute_313: "f32[128, 8]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_74: "f32[128, 128]" = torch.ops.aten.mm.default(permute_313, view_154);  permute_313 = view_154 = None
    permute_314: "f32[128, 128]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    mm_75: "f32[8, 128]" = torch.ops.aten.mm.default(view_400, permute_315);  view_400 = permute_315 = None
    view_401: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_75, [8, 1, 128]);  mm_75 = None
    permute_316: "f32[128, 128]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    slice_scatter_27: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, view_401, 1, 0, 1);  view_401 = None
    slice_scatter_28: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_27, 0, 0, 9223372036854775807);  slice_scatter_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_233: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_232, slice_scatter_28);  add_232 = slice_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_440: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_233, primals_163);  primals_163 = None
    mul_441: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_440, 128)
    sum_113: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
    mul_442: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_440, mul_182);  mul_440 = None
    sum_114: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
    mul_443: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_182, sum_114);  sum_114 = None
    sub_151: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_441, sum_113);  mul_441 = sum_113 = None
    sub_152: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_151, mul_443);  sub_151 = mul_443 = None
    div_24: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 128);  rsqrt_26 = None
    mul_444: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_24, sub_152);  div_24 = sub_152 = None
    mul_445: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_233, mul_182);  mul_182 = None
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
    sum_116: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    slice_scatter_29: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_26, 1, 0, 1);  slice_scatter_26 = None
    slice_scatter_30: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_29, 0, 0, 9223372036854775807);  slice_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_234: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_444, slice_scatter_30);  mul_444 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_79: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_234, 1, 0, 1)
    slice_80: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_234, 1, 1, 401);  add_234 = None
    slice_scatter_31: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_80, 1, 1, 9223372036854775807);  slice_80 = None
    slice_scatter_32: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_31, 0, 0, 9223372036854775807);  slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_81: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_229, 1, 0, 1)
    slice_82: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_229, 1, 1, 401);  add_229 = None
    slice_scatter_33: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_82, 1, 1, 9223372036854775807);  slice_82 = None
    slice_scatter_34: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_33, 0, 0, 9223372036854775807);  slice_scatter_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_235: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_34);  slice_scatter_32 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_402: "f32[8, 128]" = torch.ops.aten.view.default(slice_81, [8, 128]);  slice_81 = None
    mm_76: "f32[8, 256]" = torch.ops.aten.mm.default(view_402, permute_317);  permute_317 = None
    permute_318: "f32[128, 8]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_77: "f32[128, 256]" = torch.ops.aten.mm.default(permute_318, view_152);  permute_318 = view_152 = None
    permute_319: "f32[256, 128]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
    view_403: "f32[128]" = torch.ops.aten.view.default(sum_117, [128]);  sum_117 = None
    permute_320: "f32[128, 256]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_404: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_76, [8, 1, 256]);  mm_76 = None
    mul_447: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_448: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, add_135)
    mul_449: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_448, -0.5);  mul_448 = None
    exp_15: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_449);  mul_449 = None
    mul_450: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_451: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_450);  add_135 = mul_450 = None
    add_237: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_447, mul_451);  mul_447 = mul_451 = None
    mul_452: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_404, add_237);  view_404 = add_237 = None
    mul_454: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_452, primals_159);  primals_159 = None
    mul_455: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_454, 256)
    sum_118: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_454, mul_177);  mul_454 = None
    sum_119: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_177, sum_119);  sum_119 = None
    sub_154: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_455, sum_118);  mul_455 = sum_118 = None
    sub_155: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_154, mul_457);  sub_154 = mul_457 = None
    mul_458: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_155);  div_25 = sub_155 = None
    mul_459: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_452, mul_177);  mul_177 = None
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    slice_scatter_35: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_default_15, mul_458, 0, 0, 9223372036854775807);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_405: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_35, [8, 256])
    mm_78: "f32[8, 256]" = torch.ops.aten.mm.default(view_405, permute_321);  permute_321 = None
    permute_322: "f32[256, 8]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_79: "f32[256, 256]" = torch.ops.aten.mm.default(permute_322, view_150);  permute_322 = view_150 = None
    permute_323: "f32[256, 256]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
    view_406: "f32[256]" = torch.ops.aten.view.default(sum_122, [256]);  sum_122 = None
    permute_324: "f32[256, 256]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_407: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_78, [8, 1, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_408: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_407, [8, 1, 4, 64]);  view_407 = None
    permute_325: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
    view_409: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_325, [32, 1, 64]);  permute_325 = None
    bmm_24: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_326, view_409);  permute_326 = None
    bmm_25: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_409, permute_327);  view_409 = permute_327 = None
    view_410: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_24, [8, 4, 197, 64]);  bmm_24 = None
    view_411: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_25, [8, 4, 1, 197]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_460: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_411, alias_25);  view_411 = None
    sum_123: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [-1], True)
    mul_461: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_25, sum_123);  alias_25 = sum_123 = None
    sub_156: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_462: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_156, 0.125);  sub_156 = None
    view_412: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_462, [32, 1, 197]);  mul_462 = None
    bmm_26: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_328, view_412);  permute_328 = None
    bmm_27: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_412, permute_329);  view_412 = permute_329 = None
    view_413: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_26, [8, 4, 64, 197]);  bmm_26 = None
    view_414: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_27, [8, 4, 1, 64]);  bmm_27 = None
    permute_330: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_413, [0, 1, 3, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_331: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_82: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_415: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_82, [8, 197, 256]);  clone_82 = None
    view_416: "f32[1576, 256]" = torch.ops.aten.view.default(view_415, [1576, 256]);  view_415 = None
    mm_80: "f32[1576, 256]" = torch.ops.aten.mm.default(view_416, permute_332);  permute_332 = None
    permute_333: "f32[256, 1576]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_333, view_137);  permute_333 = None
    permute_334: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.view.default(sum_124, [256]);  sum_124 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_418: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_80, [8, 197, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_336: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_330, [0, 2, 1, 3]);  permute_330 = None
    view_419: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_336, [8, 197, 256]);  permute_336 = None
    clone_83: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_419, memory_format = torch.contiguous_format);  view_419 = None
    view_420: "f32[1576, 256]" = torch.ops.aten.view.default(clone_83, [1576, 256]);  clone_83 = None
    mm_82: "f32[1576, 256]" = torch.ops.aten.mm.default(view_420, permute_337);  permute_337 = None
    permute_338: "f32[256, 1576]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_338, view_137);  permute_338 = view_137 = None
    permute_339: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[256]" = torch.ops.aten.view.default(sum_125, [256]);  sum_125 = None
    permute_340: "f32[256, 256]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_422: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_82, [8, 197, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_238: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_418, view_422);  view_418 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_341: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    view_423: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_341, [8, 1, 256]);  permute_341 = None
    sum_126: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_423, [0, 1], True)
    view_424: "f32[256]" = torch.ops.aten.view.default(sum_126, [256]);  sum_126 = None
    view_425: "f32[8, 256]" = torch.ops.aten.view.default(view_423, [8, 256]);  view_423 = None
    permute_342: "f32[256, 8]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_84: "f32[256, 256]" = torch.ops.aten.mm.default(permute_342, view_134);  permute_342 = view_134 = None
    permute_343: "f32[256, 256]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    mm_85: "f32[8, 256]" = torch.ops.aten.mm.default(view_425, permute_344);  view_425 = permute_344 = None
    view_426: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_85, [8, 1, 256]);  mm_85 = None
    permute_345: "f32[256, 256]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    slice_scatter_36: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, view_426, 1, 0, 1);  view_426 = None
    slice_scatter_37: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_36, 0, 0, 9223372036854775807);  slice_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_239: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_238, slice_scatter_37);  add_238 = slice_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_464: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_239, primals_149);  primals_149 = None
    mul_465: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_464, 256)
    sum_127: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
    mul_466: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_464, mul_174);  mul_464 = None
    sum_128: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
    mul_467: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_174, sum_128);  sum_128 = None
    sub_158: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_465, sum_127);  mul_465 = sum_127 = None
    sub_159: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_158, mul_467);  sub_158 = mul_467 = None
    div_26: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    mul_468: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_26, sub_159);  div_26 = sub_159 = None
    mul_469: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_239, mul_174);  mul_174 = None
    sum_129: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    slice_scatter_38: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_35, 1, 0, 1);  slice_scatter_35 = None
    slice_scatter_39: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_38, 0, 0, 9223372036854775807);  slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_240: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_468, slice_scatter_39);  mul_468 = slice_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_83: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_240, 1, 0, 1)
    slice_84: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_240, 1, 1, 197);  add_240 = None
    slice_scatter_40: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_84, 1, 1, 9223372036854775807);  slice_84 = None
    slice_scatter_41: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_40, 0, 0, 9223372036854775807);  slice_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_241: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_25, slice_scatter_41);  slice_scatter_25 = slice_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_427: "f32[8, 128]" = torch.ops.aten.view.default(slice_79, [8, 128]);  slice_79 = None
    mm_86: "f32[8, 256]" = torch.ops.aten.mm.default(view_427, permute_346);  permute_346 = None
    permute_347: "f32[128, 8]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_87: "f32[128, 256]" = torch.ops.aten.mm.default(permute_347, view_132);  permute_347 = view_132 = None
    permute_348: "f32[256, 128]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
    view_428: "f32[128]" = torch.ops.aten.view.default(sum_131, [128]);  sum_131 = None
    permute_349: "f32[128, 256]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_429: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_86, [8, 1, 256]);  mm_86 = None
    mul_471: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_129, 0.5);  add_129 = None
    mul_472: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, add_128)
    mul_473: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_472, -0.5);  mul_472 = None
    exp_16: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_473);  mul_473 = None
    mul_474: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_475: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, mul_474);  add_128 = mul_474 = None
    add_243: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_471, mul_475);  mul_471 = mul_475 = None
    mul_476: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_429, add_243);  view_429 = add_243 = None
    mul_478: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_476, primals_145);  primals_145 = None
    mul_479: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_478, 256)
    sum_132: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True)
    mul_480: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_478, mul_169);  mul_478 = None
    sum_133: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
    mul_481: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_169, sum_133);  sum_133 = None
    sub_161: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_479, sum_132);  mul_479 = sum_132 = None
    sub_162: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_161, mul_481);  sub_161 = mul_481 = None
    mul_482: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_162);  div_27 = sub_162 = None
    mul_483: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_476, mul_169);  mul_169 = None
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1]);  mul_483 = None
    sum_135: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    slice_scatter_42: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, mul_482, 1, 0, 1);  mul_482 = None
    slice_scatter_43: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_42, 0, 0, 9223372036854775807);  slice_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_244: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_241, slice_scatter_43);  add_241 = slice_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_430: "f32[8, 256]" = torch.ops.aten.view.default(slice_83, [8, 256]);  slice_83 = None
    mm_88: "f32[8, 128]" = torch.ops.aten.mm.default(view_430, permute_350);  permute_350 = None
    permute_351: "f32[256, 8]" = torch.ops.aten.permute.default(view_430, [1, 0])
    mm_89: "f32[256, 128]" = torch.ops.aten.mm.default(permute_351, view_130);  permute_351 = view_130 = None
    permute_352: "f32[128, 256]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_136: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
    view_431: "f32[256]" = torch.ops.aten.view.default(sum_136, [256]);  sum_136 = None
    permute_353: "f32[256, 128]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_432: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_88, [8, 1, 128]);  mm_88 = None
    mul_485: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_126, 0.5);  add_126 = None
    mul_486: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, add_125)
    mul_487: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
    exp_17: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
    mul_488: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_489: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, mul_488);  add_125 = mul_488 = None
    add_246: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
    mul_490: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_432, add_246);  view_432 = add_246 = None
    mul_492: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_490, primals_141);  primals_141 = None
    mul_493: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_492, 128)
    sum_137: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
    mul_494: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_492, mul_164);  mul_492 = None
    sum_138: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
    mul_495: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_164, sum_138);  sum_138 = None
    sub_164: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_493, sum_137);  mul_493 = sum_137 = None
    sub_165: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_164, mul_495);  sub_164 = mul_495 = None
    mul_496: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_28, sub_165);  div_28 = sub_165 = None
    mul_497: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_490, mul_164);  mul_164 = None
    sum_139: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
    sum_140: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
    slice_scatter_44: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_496, 1, 0, 1);  mul_496 = None
    slice_scatter_45: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_44, 0, 0, 9223372036854775807);  slice_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_247: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_235, slice_scatter_45);  add_235 = slice_scatter_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_433: "f32[1576, 256]" = torch.ops.aten.view.default(add_244, [1576, 256])
    mm_90: "f32[1576, 768]" = torch.ops.aten.mm.default(view_433, permute_354);  permute_354 = None
    permute_355: "f32[256, 1576]" = torch.ops.aten.permute.default(view_433, [1, 0])
    mm_91: "f32[256, 768]" = torch.ops.aten.mm.default(permute_355, view_128);  permute_355 = view_128 = None
    permute_356: "f32[768, 256]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
    view_434: "f32[256]" = torch.ops.aten.view.default(sum_141, [256]);  sum_141 = None
    permute_357: "f32[256, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_435: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_90, [8, 197, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_499: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_122, 0.5);  add_122 = None
    mul_500: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, view_127)
    mul_501: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_500, -0.5);  mul_500 = None
    exp_18: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_501);  mul_501 = None
    mul_502: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_503: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_502);  view_127 = mul_502 = None
    add_249: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_499, mul_503);  mul_499 = mul_503 = None
    mul_504: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_435, add_249);  view_435 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_436: "f32[1576, 768]" = torch.ops.aten.view.default(mul_504, [1576, 768]);  mul_504 = None
    mm_92: "f32[1576, 256]" = torch.ops.aten.mm.default(view_436, permute_358);  permute_358 = None
    permute_359: "f32[768, 1576]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_93: "f32[768, 256]" = torch.ops.aten.mm.default(permute_359, view_126);  permute_359 = view_126 = None
    permute_360: "f32[256, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    permute_361: "f32[768, 256]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    view_438: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_92, [8, 197, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_506: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_438, primals_135);  primals_135 = None
    mul_507: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_506, 256)
    sum_143: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
    mul_508: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_506, mul_159);  mul_506 = None
    sum_144: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
    mul_509: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_159, sum_144);  sum_144 = None
    sub_167: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_507, sum_143);  mul_507 = sum_143 = None
    sub_168: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_167, mul_509);  sub_167 = mul_509 = None
    mul_510: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_29, sub_168);  div_29 = sub_168 = None
    mul_511: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_438, mul_159);  mul_159 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_438, [0, 1]);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_250: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_244, mul_510);  add_244 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_439: "f32[1576, 256]" = torch.ops.aten.view.default(add_250, [1576, 256])
    mm_94: "f32[1576, 256]" = torch.ops.aten.mm.default(view_439, permute_362);  permute_362 = None
    permute_363: "f32[256, 1576]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_363, view_124);  permute_363 = view_124 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_147: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[256]" = torch.ops.aten.view.default(sum_147, [256]);  sum_147 = None
    permute_365: "f32[256, 256]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_441: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_94, [8, 197, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_442: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_441, [8, 197, 4, 64]);  view_441 = None
    permute_366: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_366, getitem_126, getitem_127, getitem_128, alias_26, getitem_130, getitem_131, getitem_132, 0, 0, 0.0, False, getitem_135, getitem_136);  permute_366 = getitem_126 = getitem_127 = getitem_128 = alias_26 = getitem_130 = getitem_131 = getitem_132 = getitem_135 = getitem_136 = None
    getitem_244: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[0]
    getitem_245: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[1]
    getitem_246: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_19: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_244, getitem_245, getitem_246]);  getitem_244 = getitem_245 = getitem_246 = None
    view_443: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_19, [3, 8, 4, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_367: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_443, [1, 3, 0, 2, 4]);  view_443 = None
    clone_86: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    view_444: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_86, [8, 197, 768]);  clone_86 = None
    view_445: "f32[1576, 768]" = torch.ops.aten.view.default(view_444, [1576, 768]);  view_444 = None
    mm_96: "f32[1576, 256]" = torch.ops.aten.mm.default(view_445, permute_368);  permute_368 = None
    permute_369: "f32[768, 1576]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_97: "f32[768, 256]" = torch.ops.aten.mm.default(permute_369, view_120);  permute_369 = view_120 = None
    permute_370: "f32[256, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
    permute_371: "f32[768, 256]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_447: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_96, [8, 197, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_513: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_447, primals_129);  primals_129 = None
    mul_514: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_513, 256)
    sum_149: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True)
    mul_515: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_513, mul_157);  mul_513 = None
    sum_150: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True);  mul_515 = None
    mul_516: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_157, sum_150);  sum_150 = None
    sub_170: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_514, sum_149);  mul_514 = sum_149 = None
    sub_171: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_170, mul_516);  sub_170 = mul_516 = None
    mul_517: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_171);  div_30 = sub_171 = None
    mul_518: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_447, mul_157);  mul_157 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 1]);  mul_518 = None
    sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_447, [0, 1]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_251: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_250, mul_517);  add_250 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[1576, 256]" = torch.ops.aten.view.default(add_251, [1576, 256])
    mm_98: "f32[1576, 768]" = torch.ops.aten.mm.default(view_448, permute_372);  permute_372 = None
    permute_373: "f32[256, 1576]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_99: "f32[256, 768]" = torch.ops.aten.mm.default(permute_373, view_118);  permute_373 = view_118 = None
    permute_374: "f32[768, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_153: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[256]" = torch.ops.aten.view.default(sum_153, [256]);  sum_153 = None
    permute_375: "f32[256, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_450: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_98, [8, 197, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_520: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_521: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, view_117)
    mul_522: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_521, -0.5);  mul_521 = None
    exp_19: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_522);  mul_522 = None
    mul_523: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_524: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, mul_523);  view_117 = mul_523 = None
    add_253: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_520, mul_524);  mul_520 = mul_524 = None
    mul_525: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_450, add_253);  view_450 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_451: "f32[1576, 768]" = torch.ops.aten.view.default(mul_525, [1576, 768]);  mul_525 = None
    mm_100: "f32[1576, 256]" = torch.ops.aten.mm.default(view_451, permute_376);  permute_376 = None
    permute_377: "f32[768, 1576]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_101: "f32[768, 256]" = torch.ops.aten.mm.default(permute_377, view_116);  permute_377 = view_116 = None
    permute_378: "f32[256, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_379: "f32[768, 256]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_453: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_100, [8, 197, 256]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_527: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_453, primals_123);  primals_123 = None
    mul_528: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_527, 256)
    sum_155: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
    mul_529: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_527, mul_152);  mul_527 = None
    sum_156: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
    mul_530: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_152, sum_156);  sum_156 = None
    sub_173: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_528, sum_155);  mul_528 = sum_155 = None
    sub_174: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_173, mul_530);  sub_173 = mul_530 = None
    mul_531: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_174);  div_31 = sub_174 = None
    mul_532: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_453, mul_152);  mul_152 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_453, [0, 1]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_254: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_251, mul_531);  add_251 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_454: "f32[1576, 256]" = torch.ops.aten.view.default(add_254, [1576, 256])
    mm_102: "f32[1576, 256]" = torch.ops.aten.mm.default(view_454, permute_380);  permute_380 = None
    permute_381: "f32[256, 1576]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_103: "f32[256, 256]" = torch.ops.aten.mm.default(permute_381, view_114);  permute_381 = view_114 = None
    permute_382: "f32[256, 256]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_159: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[256]" = torch.ops.aten.view.default(sum_159, [256]);  sum_159 = None
    permute_383: "f32[256, 256]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_456: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_102, [8, 197, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_457: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_456, [8, 197, 4, 64]);  view_456 = None
    permute_384: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_384, getitem_110, getitem_111, getitem_112, alias_27, getitem_114, getitem_115, getitem_116, 0, 0, 0.0, False, getitem_119, getitem_120);  permute_384 = getitem_110 = getitem_111 = getitem_112 = alias_27 = getitem_114 = getitem_115 = getitem_116 = getitem_119 = getitem_120 = None
    getitem_247: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[0]
    getitem_248: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[1]
    getitem_249: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_20: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_247, getitem_248, getitem_249]);  getitem_247 = getitem_248 = getitem_249 = None
    view_458: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_20, [3, 8, 4, 197, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_385: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_458, [1, 3, 0, 2, 4]);  view_458 = None
    clone_87: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_459: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_87, [8, 197, 768]);  clone_87 = None
    view_460: "f32[1576, 768]" = torch.ops.aten.view.default(view_459, [1576, 768]);  view_459 = None
    mm_104: "f32[1576, 256]" = torch.ops.aten.mm.default(view_460, permute_386);  permute_386 = None
    permute_387: "f32[768, 1576]" = torch.ops.aten.permute.default(view_460, [1, 0])
    mm_105: "f32[768, 256]" = torch.ops.aten.mm.default(permute_387, view_110);  permute_387 = view_110 = None
    permute_388: "f32[256, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_460, [0], True);  view_460 = None
    view_461: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    permute_389: "f32[768, 256]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    view_462: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_104, [8, 197, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_534: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_462, primals_117);  primals_117 = None
    mul_535: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_534, 256)
    sum_161: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True)
    mul_536: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_534, mul_150);  mul_534 = None
    sum_162: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True);  mul_536 = None
    mul_537: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_150, sum_162);  sum_162 = None
    sub_176: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_535, sum_161);  mul_535 = sum_161 = None
    sub_177: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_176, mul_537);  sub_176 = mul_537 = None
    mul_538: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_32, sub_177);  div_32 = sub_177 = None
    mul_539: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_462, mul_150);  mul_150 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 1]);  mul_539 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_462, [0, 1]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_255: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_254, mul_538);  add_254 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_463: "f32[1576, 256]" = torch.ops.aten.view.default(add_255, [1576, 256])
    mm_106: "f32[1576, 768]" = torch.ops.aten.mm.default(view_463, permute_390);  permute_390 = None
    permute_391: "f32[256, 1576]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_107: "f32[256, 768]" = torch.ops.aten.mm.default(permute_391, view_108);  permute_391 = view_108 = None
    permute_392: "f32[768, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_165: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[256]" = torch.ops.aten.view.default(sum_165, [256]);  sum_165 = None
    permute_393: "f32[256, 768]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_465: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_106, [8, 197, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_541: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, 0.5);  add_108 = None
    mul_542: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_543: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_542, -0.5);  mul_542 = None
    exp_20: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_543);  mul_543 = None
    mul_544: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_545: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, mul_544);  view_107 = mul_544 = None
    add_257: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_541, mul_545);  mul_541 = mul_545 = None
    mul_546: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_465, add_257);  view_465 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_466: "f32[1576, 768]" = torch.ops.aten.view.default(mul_546, [1576, 768]);  mul_546 = None
    mm_108: "f32[1576, 256]" = torch.ops.aten.mm.default(view_466, permute_394);  permute_394 = None
    permute_395: "f32[768, 1576]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_109: "f32[768, 256]" = torch.ops.aten.mm.default(permute_395, view_106);  permute_395 = view_106 = None
    permute_396: "f32[256, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    permute_397: "f32[768, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_468: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_108, [8, 197, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_548: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_468, primals_111);  primals_111 = None
    mul_549: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_548, 256)
    sum_167: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True)
    mul_550: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_548, mul_145);  mul_548 = None
    sum_168: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True);  mul_550 = None
    mul_551: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_145, sum_168);  sum_168 = None
    sub_179: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_549, sum_167);  mul_549 = sum_167 = None
    sub_180: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_179, mul_551);  sub_179 = mul_551 = None
    mul_552: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_180);  div_33 = sub_180 = None
    mul_553: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_468, mul_145);  mul_145 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 1]);  mul_553 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_468, [0, 1]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_258: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_255, mul_552);  add_255 = mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_469: "f32[1576, 256]" = torch.ops.aten.view.default(add_258, [1576, 256])
    mm_110: "f32[1576, 256]" = torch.ops.aten.mm.default(view_469, permute_398);  permute_398 = None
    permute_399: "f32[256, 1576]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_111: "f32[256, 256]" = torch.ops.aten.mm.default(permute_399, view_104);  permute_399 = view_104 = None
    permute_400: "f32[256, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_471: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_110, [8, 197, 256]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_472: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_471, [8, 197, 4, 64]);  view_471 = None
    permute_402: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_402, getitem_94, getitem_95, getitem_96, alias_28, getitem_98, getitem_99, getitem_100, 0, 0, 0.0, False, getitem_103, getitem_104);  permute_402 = getitem_94 = getitem_95 = getitem_96 = alias_28 = getitem_98 = getitem_99 = getitem_100 = getitem_103 = getitem_104 = None
    getitem_250: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[0]
    getitem_251: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[1]
    getitem_252: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_21: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_250, getitem_251, getitem_252]);  getitem_250 = getitem_251 = getitem_252 = None
    view_473: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 4, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_403: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_473, [1, 3, 0, 2, 4]);  view_473 = None
    clone_88: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
    view_474: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_88, [8, 197, 768]);  clone_88 = None
    view_475: "f32[1576, 768]" = torch.ops.aten.view.default(view_474, [1576, 768]);  view_474 = None
    mm_112: "f32[1576, 256]" = torch.ops.aten.mm.default(view_475, permute_404);  permute_404 = None
    permute_405: "f32[768, 1576]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_113: "f32[768, 256]" = torch.ops.aten.mm.default(permute_405, view_100);  permute_405 = view_100 = None
    permute_406: "f32[256, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_407: "f32[768, 256]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_477: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_112, [8, 197, 256]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_555: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_477, primals_105);  primals_105 = None
    mul_556: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_555, 256)
    sum_173: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True)
    mul_557: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_555, mul_143);  mul_555 = None
    sum_174: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
    mul_558: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_143, sum_174);  sum_174 = None
    sub_182: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_556, sum_173);  mul_556 = sum_173 = None
    sub_183: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_182, mul_558);  sub_182 = mul_558 = None
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    mul_559: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_183);  div_34 = sub_183 = None
    mul_560: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_477, mul_143);  mul_143 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 1]);  mul_560 = None
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_477, [0, 1]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_259: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_258, mul_559);  add_258 = mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_478: "f32[3208, 128]" = torch.ops.aten.view.default(add_247, [3208, 128])
    mm_114: "f32[3208, 384]" = torch.ops.aten.mm.default(view_478, permute_408);  permute_408 = None
    permute_409: "f32[128, 3208]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_115: "f32[128, 384]" = torch.ops.aten.mm.default(permute_409, view_98);  permute_409 = view_98 = None
    permute_410: "f32[384, 128]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_177: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[128]" = torch.ops.aten.view.default(sum_177, [128]);  sum_177 = None
    permute_411: "f32[128, 384]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_480: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_114, [8, 401, 384]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_562: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
    mul_563: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, view_97)
    mul_564: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_563, -0.5);  mul_563 = None
    exp_21: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_564);  mul_564 = None
    mul_565: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_566: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, mul_565);  view_97 = mul_565 = None
    add_261: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_562, mul_566);  mul_562 = mul_566 = None
    mul_567: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_480, add_261);  view_480 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_481: "f32[3208, 384]" = torch.ops.aten.view.default(mul_567, [3208, 384]);  mul_567 = None
    mm_116: "f32[3208, 128]" = torch.ops.aten.mm.default(view_481, permute_412);  permute_412 = None
    permute_413: "f32[384, 3208]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_117: "f32[384, 128]" = torch.ops.aten.mm.default(permute_413, view_96);  permute_413 = view_96 = None
    permute_414: "f32[128, 384]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_178: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[384]" = torch.ops.aten.view.default(sum_178, [384]);  sum_178 = None
    permute_415: "f32[384, 128]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    view_483: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_116, [8, 401, 128]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_569: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_483, primals_99);  primals_99 = None
    mul_570: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_569, 128)
    sum_179: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True)
    mul_571: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_569, mul_138);  mul_569 = None
    sum_180: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True);  mul_571 = None
    mul_572: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_138, sum_180);  sum_180 = None
    sub_185: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_570, sum_179);  mul_570 = sum_179 = None
    sub_186: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_185, mul_572);  sub_185 = mul_572 = None
    mul_573: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_35, sub_186);  div_35 = sub_186 = None
    mul_574: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_483, mul_138);  mul_138 = None
    sum_181: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 1]);  mul_574 = None
    sum_182: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_483, [0, 1]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_262: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_247, mul_573);  add_247 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_484: "f32[3208, 128]" = torch.ops.aten.view.default(add_262, [3208, 128])
    mm_118: "f32[3208, 128]" = torch.ops.aten.mm.default(view_484, permute_416);  permute_416 = None
    permute_417: "f32[128, 3208]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_119: "f32[128, 128]" = torch.ops.aten.mm.default(permute_417, view_94);  permute_417 = view_94 = None
    permute_418: "f32[128, 128]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_183: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[128]" = torch.ops.aten.view.default(sum_183, [128]);  sum_183 = None
    permute_419: "f32[128, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    view_486: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_118, [8, 401, 128]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_487: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_486, [8, 401, 4, 32]);  view_486 = None
    permute_420: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_420, getitem_78, getitem_79, getitem_80, alias_29, getitem_82, getitem_83, getitem_84, 0, 0, 0.0, False, getitem_87, getitem_88);  permute_420 = getitem_78 = getitem_79 = getitem_80 = alias_29 = getitem_82 = getitem_83 = getitem_84 = getitem_87 = getitem_88 = None
    getitem_253: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[0]
    getitem_254: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[1]
    getitem_255: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_22: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_253, getitem_254, getitem_255]);  getitem_253 = getitem_254 = getitem_255 = None
    view_488: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_22, [3, 8, 4, 401, 32]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_421: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_488, [1, 3, 0, 2, 4]);  view_488 = None
    clone_89: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_489: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_89, [8, 401, 384]);  clone_89 = None
    view_490: "f32[3208, 384]" = torch.ops.aten.view.default(view_489, [3208, 384]);  view_489 = None
    mm_120: "f32[3208, 128]" = torch.ops.aten.mm.default(view_490, permute_422);  permute_422 = None
    permute_423: "f32[384, 3208]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_121: "f32[384, 128]" = torch.ops.aten.mm.default(permute_423, view_90);  permute_423 = view_90 = None
    permute_424: "f32[128, 384]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_425: "f32[384, 128]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_492: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_120, [8, 401, 128]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_576: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_492, primals_93);  primals_93 = None
    mul_577: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_576, 128)
    sum_185: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True)
    mul_578: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_576, mul_136);  mul_576 = None
    sum_186: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_578, [2], True);  mul_578 = None
    mul_579: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_136, sum_186);  sum_186 = None
    sub_188: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_577, sum_185);  mul_577 = sum_185 = None
    sub_189: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_188, mul_579);  sub_188 = mul_579 = None
    div_36: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 128);  rsqrt_14 = None
    mul_580: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_36, sub_189);  div_36 = sub_189 = None
    mul_581: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_492, mul_136);  mul_136 = None
    sum_187: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 1]);  mul_581 = None
    sum_188: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_492, [0, 1]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_263: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_262, mul_580);  add_262 = mul_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_85: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_259, 1, 0, 1)
    slice_86: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_259, 1, 1, 197);  add_259 = None
    slice_scatter_46: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_86, 1, 1, 9223372036854775807);  slice_86 = None
    slice_scatter_47: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_46, 0, 0, 9223372036854775807);  slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_493: "f32[8, 256]" = torch.ops.aten.view.default(slice_85, [8, 256]);  slice_85 = None
    mm_122: "f32[8, 128]" = torch.ops.aten.mm.default(view_493, permute_426);  permute_426 = None
    permute_427: "f32[256, 8]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_123: "f32[256, 128]" = torch.ops.aten.mm.default(permute_427, view_88);  permute_427 = view_88 = None
    permute_428: "f32[128, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_189: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[256]" = torch.ops.aten.view.default(sum_189, [256]);  sum_189 = None
    permute_429: "f32[256, 128]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_495: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_122, [8, 1, 128]);  mm_122 = None
    mul_583: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_95, 0.5);  add_95 = None
    mul_584: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, add_94)
    mul_585: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_584, -0.5);  mul_584 = None
    exp_22: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_585);  mul_585 = None
    mul_586: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_587: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, mul_586);  add_94 = mul_586 = None
    add_265: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_583, mul_587);  mul_583 = mul_587 = None
    mul_588: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_495, add_265);  view_495 = add_265 = None
    mul_590: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_588, primals_89);  primals_89 = None
    mul_591: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_590, 128)
    sum_190: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True)
    mul_592: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_590, mul_131);  mul_590 = None
    sum_191: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [2], True);  mul_592 = None
    mul_593: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_131, sum_191);  sum_191 = None
    sub_191: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_591, sum_190);  mul_591 = sum_190 = None
    sub_192: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_191, mul_593);  sub_191 = mul_593 = None
    mul_594: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_37, sub_192);  div_37 = sub_192 = None
    mul_595: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_588, mul_131);  mul_131 = None
    sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1]);  mul_595 = None
    sum_193: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_588, [0, 1]);  mul_588 = None
    slice_scatter_48: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_default_6, mul_594, 0, 0, 9223372036854775807);  full_default_6 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_496: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_48, [8, 128])
    mm_124: "f32[8, 128]" = torch.ops.aten.mm.default(view_496, permute_430);  permute_430 = None
    permute_431: "f32[128, 8]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_125: "f32[128, 128]" = torch.ops.aten.mm.default(permute_431, view_86);  permute_431 = view_86 = None
    permute_432: "f32[128, 128]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_194: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[128]" = torch.ops.aten.view.default(sum_194, [128]);  sum_194 = None
    permute_433: "f32[128, 128]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_498: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_124, [8, 1, 128]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_499: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_498, [8, 1, 4, 32]);  view_498 = None
    permute_434: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    view_500: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_434, [32, 1, 32]);  permute_434 = None
    bmm_28: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_435, view_500);  permute_435 = None
    bmm_29: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_500, permute_436);  view_500 = permute_436 = None
    view_501: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_28, [8, 4, 401, 32]);  bmm_28 = None
    view_502: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_29, [8, 4, 1, 401]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_596: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_502, alias_30);  view_502 = None
    sum_195: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [-1], True)
    mul_597: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_30, sum_195);  alias_30 = sum_195 = None
    sub_193: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_598: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_193, 0.1767766952966369);  sub_193 = None
    view_503: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_598, [32, 1, 401]);  mul_598 = None
    bmm_30: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_437, view_503);  permute_437 = None
    bmm_31: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_503, permute_438);  view_503 = permute_438 = None
    view_504: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_30, [8, 4, 32, 401]);  bmm_30 = None
    view_505: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_31, [8, 4, 1, 32]);  bmm_31 = None
    permute_439: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 2]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_440: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_90: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
    view_506: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_90, [8, 401, 128]);  clone_90 = None
    view_507: "f32[3208, 128]" = torch.ops.aten.view.default(view_506, [3208, 128]);  view_506 = None
    mm_126: "f32[3208, 128]" = torch.ops.aten.mm.default(view_507, permute_441);  permute_441 = None
    permute_442: "f32[128, 3208]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_127: "f32[128, 128]" = torch.ops.aten.mm.default(permute_442, view_73);  permute_442 = None
    permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_196: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[128]" = torch.ops.aten.view.default(sum_196, [128]);  sum_196 = None
    permute_444: "f32[128, 128]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    view_509: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_126, [8, 401, 128]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_445: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_439, [0, 2, 1, 3]);  permute_439 = None
    view_510: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_445, [8, 401, 128]);  permute_445 = None
    clone_91: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_510, memory_format = torch.contiguous_format);  view_510 = None
    view_511: "f32[3208, 128]" = torch.ops.aten.view.default(clone_91, [3208, 128]);  clone_91 = None
    mm_128: "f32[3208, 128]" = torch.ops.aten.mm.default(view_511, permute_446);  permute_446 = None
    permute_447: "f32[128, 3208]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_129: "f32[128, 128]" = torch.ops.aten.mm.default(permute_447, view_73);  permute_447 = view_73 = None
    permute_448: "f32[128, 128]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_197: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[128]" = torch.ops.aten.view.default(sum_197, [128]);  sum_197 = None
    permute_449: "f32[128, 128]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_513: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_128, [8, 401, 128]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_266: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_509, view_513);  view_509 = view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_450: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
    view_514: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_450, [8, 1, 128]);  permute_450 = None
    sum_198: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_514, [0, 1], True)
    view_515: "f32[128]" = torch.ops.aten.view.default(sum_198, [128]);  sum_198 = None
    view_516: "f32[8, 128]" = torch.ops.aten.view.default(view_514, [8, 128]);  view_514 = None
    permute_451: "f32[128, 8]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_130: "f32[128, 128]" = torch.ops.aten.mm.default(permute_451, view_70);  permute_451 = view_70 = None
    permute_452: "f32[128, 128]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    mm_131: "f32[8, 128]" = torch.ops.aten.mm.default(view_516, permute_453);  view_516 = permute_453 = None
    view_517: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_131, [8, 1, 128]);  mm_131 = None
    permute_454: "f32[128, 128]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    slice_scatter_49: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, view_517, 1, 0, 1);  view_517 = None
    slice_scatter_50: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_49, 0, 0, 9223372036854775807);  slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_267: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_266, slice_scatter_50);  add_266 = slice_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_600: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_267, primals_79);  primals_79 = None
    mul_601: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_600, 128)
    sum_199: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
    mul_602: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_600, mul_128);  mul_600 = None
    sum_200: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
    mul_603: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_128, sum_200);  sum_200 = None
    sub_195: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_601, sum_199);  mul_601 = sum_199 = None
    sub_196: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_195, mul_603);  sub_195 = mul_603 = None
    div_38: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 128);  rsqrt_12 = None
    mul_604: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_38, sub_196);  div_38 = sub_196 = None
    mul_605: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_267, mul_128);  mul_128 = None
    sum_201: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
    slice_scatter_51: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_48, 1, 0, 1);  slice_scatter_48 = None
    slice_scatter_52: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_51, 0, 0, 9223372036854775807);  slice_scatter_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_268: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_604, slice_scatter_52);  mul_604 = slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_87: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_268, 1, 0, 1)
    slice_88: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_268, 1, 1, 401);  add_268 = None
    slice_scatter_53: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_88, 1, 1, 9223372036854775807);  slice_88 = None
    slice_scatter_54: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_53, 0, 0, 9223372036854775807);  slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_89: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_263, 1, 0, 1)
    slice_90: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_263, 1, 1, 401);  add_263 = None
    slice_scatter_55: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_90, 1, 1, 9223372036854775807);  slice_90 = None
    slice_scatter_56: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_55, 0, 0, 9223372036854775807);  slice_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_269: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_56);  slice_scatter_54 = slice_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_518: "f32[8, 128]" = torch.ops.aten.view.default(slice_89, [8, 128]);  slice_89 = None
    mm_132: "f32[8, 256]" = torch.ops.aten.mm.default(view_518, permute_455);  permute_455 = None
    permute_456: "f32[128, 8]" = torch.ops.aten.permute.default(view_518, [1, 0])
    mm_133: "f32[128, 256]" = torch.ops.aten.mm.default(permute_456, view_68);  permute_456 = view_68 = None
    permute_457: "f32[256, 128]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_203: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
    view_519: "f32[128]" = torch.ops.aten.view.default(sum_203, [128]);  sum_203 = None
    permute_458: "f32[128, 256]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_520: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_132, [8, 1, 256]);  mm_132 = None
    mul_607: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_608: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, add_87)
    mul_609: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_608, -0.5);  mul_608 = None
    exp_23: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_609);  mul_609 = None
    mul_610: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_611: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, mul_610);  add_87 = mul_610 = None
    add_271: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_607, mul_611);  mul_607 = mul_611 = None
    mul_612: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_520, add_271);  view_520 = add_271 = None
    mul_614: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_612, primals_75);  primals_75 = None
    mul_615: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_614, 256)
    sum_204: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [2], True)
    mul_616: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_614, mul_123);  mul_614 = None
    sum_205: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [2], True);  mul_616 = None
    mul_617: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_123, sum_205);  sum_205 = None
    sub_198: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_615, sum_204);  mul_615 = sum_204 = None
    sub_199: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_198, mul_617);  sub_198 = mul_617 = None
    mul_618: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_199);  div_39 = sub_199 = None
    mul_619: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_612, mul_123);  mul_123 = None
    sum_206: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 1]);  mul_619 = None
    sum_207: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
    slice_scatter_57: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_default_15, mul_618, 0, 0, 9223372036854775807);  full_default_15 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_521: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_57, [8, 256])
    mm_134: "f32[8, 256]" = torch.ops.aten.mm.default(view_521, permute_459);  permute_459 = None
    permute_460: "f32[256, 8]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_135: "f32[256, 256]" = torch.ops.aten.mm.default(permute_460, view_66);  permute_460 = view_66 = None
    permute_461: "f32[256, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_208: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[256]" = torch.ops.aten.view.default(sum_208, [256]);  sum_208 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_523: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_134, [8, 1, 256]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_524: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_523, [8, 1, 4, 64]);  view_523 = None
    permute_463: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    view_525: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_463, [32, 1, 64]);  permute_463 = None
    bmm_32: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_464, view_525);  permute_464 = None
    bmm_33: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_525, permute_465);  view_525 = permute_465 = None
    view_526: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_32, [8, 4, 197, 64]);  bmm_32 = None
    view_527: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_33, [8, 4, 1, 197]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    mul_620: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_527, alias_31);  view_527 = None
    sum_209: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [-1], True)
    mul_621: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_31, sum_209);  alias_31 = sum_209 = None
    sub_200: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_622: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_200, 0.125);  sub_200 = None
    view_528: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_622, [32, 1, 197]);  mul_622 = None
    bmm_34: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_466, view_528);  permute_466 = None
    bmm_35: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_528, permute_467);  view_528 = permute_467 = None
    view_529: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_34, [8, 4, 64, 197]);  bmm_34 = None
    view_530: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_35, [8, 4, 1, 64]);  bmm_35 = None
    permute_468: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_469: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
    clone_92: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
    view_531: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_92, [8, 197, 256]);  clone_92 = None
    view_532: "f32[1576, 256]" = torch.ops.aten.view.default(view_531, [1576, 256]);  view_531 = None
    mm_136: "f32[1576, 256]" = torch.ops.aten.mm.default(view_532, permute_470);  permute_470 = None
    permute_471: "f32[256, 1576]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_137: "f32[256, 256]" = torch.ops.aten.mm.default(permute_471, view_53);  permute_471 = None
    permute_472: "f32[256, 256]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_210: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[256]" = torch.ops.aten.view.default(sum_210, [256]);  sum_210 = None
    permute_473: "f32[256, 256]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_534: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_136, [8, 197, 256]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_474: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_468, [0, 2, 1, 3]);  permute_468 = None
    view_535: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_474, [8, 197, 256]);  permute_474 = None
    clone_93: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_535, memory_format = torch.contiguous_format);  view_535 = None
    view_536: "f32[1576, 256]" = torch.ops.aten.view.default(clone_93, [1576, 256]);  clone_93 = None
    mm_138: "f32[1576, 256]" = torch.ops.aten.mm.default(view_536, permute_475);  permute_475 = None
    permute_476: "f32[256, 1576]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_139: "f32[256, 256]" = torch.ops.aten.mm.default(permute_476, view_53);  permute_476 = view_53 = None
    permute_477: "f32[256, 256]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_211: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[256]" = torch.ops.aten.view.default(sum_211, [256]);  sum_211 = None
    permute_478: "f32[256, 256]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_538: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_138, [8, 197, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_272: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_534, view_538);  view_534 = view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_479: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    view_539: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_479, [8, 1, 256]);  permute_479 = None
    sum_212: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_539, [0, 1], True)
    view_540: "f32[256]" = torch.ops.aten.view.default(sum_212, [256]);  sum_212 = None
    view_541: "f32[8, 256]" = torch.ops.aten.view.default(view_539, [8, 256]);  view_539 = None
    permute_480: "f32[256, 8]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_140: "f32[256, 256]" = torch.ops.aten.mm.default(permute_480, view_50);  permute_480 = view_50 = None
    permute_481: "f32[256, 256]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    mm_141: "f32[8, 256]" = torch.ops.aten.mm.default(view_541, permute_482);  view_541 = permute_482 = None
    view_542: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_141, [8, 1, 256]);  mm_141 = None
    permute_483: "f32[256, 256]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    slice_scatter_58: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, view_542, 1, 0, 1);  view_542 = None
    slice_scatter_59: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_58, 0, 0, 9223372036854775807);  slice_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_273: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_272, slice_scatter_59);  add_272 = slice_scatter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    mul_624: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_273, primals_65);  primals_65 = None
    mul_625: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_624, 256)
    sum_213: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True)
    mul_626: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_624, mul_120);  mul_624 = None
    sum_214: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True);  mul_626 = None
    mul_627: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_120, sum_214);  sum_214 = None
    sub_202: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_625, sum_213);  mul_625 = sum_213 = None
    sub_203: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_202, mul_627);  sub_202 = mul_627 = None
    div_40: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    mul_628: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_203);  div_40 = sub_203 = None
    mul_629: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_273, mul_120);  mul_120 = None
    sum_215: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1]);  mul_629 = None
    sum_216: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 1]);  add_273 = None
    slice_scatter_60: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_57, 1, 0, 1);  slice_scatter_57 = None
    slice_scatter_61: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_60, 0, 0, 9223372036854775807);  slice_scatter_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_274: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_628, slice_scatter_61);  mul_628 = slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_91: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_274, 1, 0, 1)
    slice_92: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_274, 1, 1, 197);  add_274 = None
    slice_scatter_62: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_92, 1, 1, 9223372036854775807);  slice_92 = None
    slice_scatter_63: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_62, 0, 0, 9223372036854775807);  slice_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_275: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_47, slice_scatter_63);  slice_scatter_47 = slice_scatter_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_543: "f32[8, 128]" = torch.ops.aten.view.default(slice_87, [8, 128]);  slice_87 = None
    mm_142: "f32[8, 256]" = torch.ops.aten.mm.default(view_543, permute_484);  permute_484 = None
    permute_485: "f32[128, 8]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_143: "f32[128, 256]" = torch.ops.aten.mm.default(permute_485, view_48);  permute_485 = view_48 = None
    permute_486: "f32[256, 128]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_217: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[128]" = torch.ops.aten.view.default(sum_217, [128]);  sum_217 = None
    permute_487: "f32[128, 256]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    view_545: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_142, [8, 1, 256]);  mm_142 = None
    mul_631: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_81, 0.5);  add_81 = None
    mul_632: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, add_80)
    mul_633: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_632, -0.5);  mul_632 = None
    exp_24: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_633);  mul_633 = None
    mul_634: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_635: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, mul_634);  add_80 = mul_634 = None
    add_277: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_631, mul_635);  mul_631 = mul_635 = None
    mul_636: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_545, add_277);  view_545 = add_277 = None
    mul_638: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_636, primals_61);  primals_61 = None
    mul_639: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_638, 256)
    sum_218: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True)
    mul_640: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_638, mul_115);  mul_638 = None
    sum_219: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [2], True);  mul_640 = None
    mul_641: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_115, sum_219);  sum_219 = None
    sub_205: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_639, sum_218);  mul_639 = sum_218 = None
    sub_206: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_205, mul_641);  sub_205 = mul_641 = None
    mul_642: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_41, sub_206);  div_41 = sub_206 = None
    mul_643: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_636, mul_115);  mul_115 = None
    sum_220: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
    sum_221: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1]);  mul_636 = None
    slice_scatter_64: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, mul_642, 1, 0, 1);  mul_642 = None
    slice_scatter_65: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_64, 0, 0, 9223372036854775807);  full_default = slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_278: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_275, slice_scatter_65);  add_275 = slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_546: "f32[8, 256]" = torch.ops.aten.view.default(slice_91, [8, 256]);  slice_91 = None
    mm_144: "f32[8, 128]" = torch.ops.aten.mm.default(view_546, permute_488);  permute_488 = None
    permute_489: "f32[256, 8]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_145: "f32[256, 128]" = torch.ops.aten.mm.default(permute_489, view_46);  permute_489 = view_46 = None
    permute_490: "f32[128, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_222: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
    view_547: "f32[256]" = torch.ops.aten.view.default(sum_222, [256]);  sum_222 = None
    permute_491: "f32[256, 128]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_548: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_144, [8, 1, 128]);  mm_144 = None
    mul_645: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_646: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, add_77)
    mul_647: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_646, -0.5);  mul_646 = None
    exp_25: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_647);  mul_647 = None
    mul_648: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_649: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, mul_648);  add_77 = mul_648 = None
    add_280: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_645, mul_649);  mul_645 = mul_649 = None
    mul_650: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_548, add_280);  view_548 = add_280 = None
    mul_652: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_650, primals_57);  primals_57 = None
    mul_653: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_652, 128)
    sum_223: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [2], True)
    mul_654: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_652, mul_110);  mul_652 = None
    sum_224: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_654, [2], True);  mul_654 = None
    mul_655: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_110, sum_224);  sum_224 = None
    sub_208: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_653, sum_223);  mul_653 = sum_223 = None
    sub_209: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_208, mul_655);  sub_208 = mul_655 = None
    mul_656: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_42, sub_209);  div_42 = sub_209 = None
    mul_657: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_650, mul_110);  mul_110 = None
    sum_225: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 1]);  mul_657 = None
    sum_226: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
    slice_scatter_66: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_656, 1, 0, 1);  mul_656 = None
    slice_scatter_67: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_default_2, slice_scatter_66, 0, 0, 9223372036854775807);  full_default_2 = slice_scatter_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_281: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_269, slice_scatter_67);  add_269 = slice_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_549: "f32[1576, 256]" = torch.ops.aten.view.default(add_278, [1576, 256])
    mm_146: "f32[1576, 768]" = torch.ops.aten.mm.default(view_549, permute_492);  permute_492 = None
    permute_493: "f32[256, 1576]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_147: "f32[256, 768]" = torch.ops.aten.mm.default(permute_493, view_44);  permute_493 = view_44 = None
    permute_494: "f32[768, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_227: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[256]" = torch.ops.aten.view.default(sum_227, [256]);  sum_227 = None
    permute_495: "f32[256, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_551: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_146, [8, 197, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_659: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
    mul_660: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_661: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_660, -0.5);  mul_660 = None
    exp_26: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_661);  mul_661 = None
    mul_662: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_663: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, mul_662);  view_43 = mul_662 = None
    add_283: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_659, mul_663);  mul_659 = mul_663 = None
    mul_664: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_551, add_283);  view_551 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_552: "f32[1576, 768]" = torch.ops.aten.view.default(mul_664, [1576, 768]);  mul_664 = None
    mm_148: "f32[1576, 256]" = torch.ops.aten.mm.default(view_552, permute_496);  permute_496 = None
    permute_497: "f32[768, 1576]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_149: "f32[768, 256]" = torch.ops.aten.mm.default(permute_497, view_42);  permute_497 = view_42 = None
    permute_498: "f32[256, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_228: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[768]" = torch.ops.aten.view.default(sum_228, [768]);  sum_228 = None
    permute_499: "f32[768, 256]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_554: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_148, [8, 197, 256]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_666: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_554, primals_51);  primals_51 = None
    mul_667: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_666, 256)
    sum_229: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True)
    mul_668: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_666, mul_105);  mul_666 = None
    sum_230: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [2], True);  mul_668 = None
    mul_669: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_105, sum_230);  sum_230 = None
    sub_211: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_667, sum_229);  mul_667 = sum_229 = None
    sub_212: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_211, mul_669);  sub_211 = mul_669 = None
    mul_670: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_212);  div_43 = sub_212 = None
    mul_671: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_554, mul_105);  mul_105 = None
    sum_231: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1]);  mul_671 = None
    sum_232: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_554, [0, 1]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_284: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_278, mul_670);  add_278 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_555: "f32[1576, 256]" = torch.ops.aten.view.default(add_284, [1576, 256])
    mm_150: "f32[1576, 256]" = torch.ops.aten.mm.default(view_555, permute_500);  permute_500 = None
    permute_501: "f32[256, 1576]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_151: "f32[256, 256]" = torch.ops.aten.mm.default(permute_501, view_40);  permute_501 = view_40 = None
    permute_502: "f32[256, 256]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_233: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[256]" = torch.ops.aten.view.default(sum_233, [256]);  sum_233 = None
    permute_503: "f32[256, 256]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_557: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_150, [8, 197, 256]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_558: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_557, [8, 197, 4, 64]);  view_557 = None
    permute_504: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_504, getitem_50, getitem_51, getitem_52, alias_32, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60);  permute_504 = getitem_50 = getitem_51 = getitem_52 = alias_32 = getitem_54 = getitem_55 = getitem_56 = getitem_59 = getitem_60 = None
    getitem_256: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[0]
    getitem_257: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[1]
    getitem_258: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_23: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_256, getitem_257, getitem_258]);  getitem_256 = getitem_257 = getitem_258 = None
    view_559: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 4, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_505: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_559, [1, 3, 0, 2, 4]);  view_559 = None
    clone_96: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_505, memory_format = torch.contiguous_format);  permute_505 = None
    view_560: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_96, [8, 197, 768]);  clone_96 = None
    view_561: "f32[1576, 768]" = torch.ops.aten.view.default(view_560, [1576, 768]);  view_560 = None
    mm_152: "f32[1576, 256]" = torch.ops.aten.mm.default(view_561, permute_506);  permute_506 = None
    permute_507: "f32[768, 1576]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_153: "f32[768, 256]" = torch.ops.aten.mm.default(permute_507, view_36);  permute_507 = view_36 = None
    permute_508: "f32[256, 768]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_234: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[768]" = torch.ops.aten.view.default(sum_234, [768]);  sum_234 = None
    permute_509: "f32[768, 256]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_563: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_152, [8, 197, 256]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_673: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_563, primals_45);  primals_45 = None
    mul_674: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_673, 256)
    sum_235: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True)
    mul_675: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_673, mul_103);  mul_673 = None
    sum_236: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True);  mul_675 = None
    mul_676: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_103, sum_236);  sum_236 = None
    sub_214: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_674, sum_235);  mul_674 = sum_235 = None
    sub_215: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_214, mul_676);  sub_214 = mul_676 = None
    mul_677: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_44, sub_215);  div_44 = sub_215 = None
    mul_678: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_563, mul_103);  mul_103 = None
    sum_237: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 1]);  mul_678 = None
    sum_238: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_285: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_284, mul_677);  add_284 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_564: "f32[1576, 256]" = torch.ops.aten.view.default(add_285, [1576, 256])
    mm_154: "f32[1576, 768]" = torch.ops.aten.mm.default(view_564, permute_510);  permute_510 = None
    permute_511: "f32[256, 1576]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_155: "f32[256, 768]" = torch.ops.aten.mm.default(permute_511, view_34);  permute_511 = view_34 = None
    permute_512: "f32[768, 256]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_239: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[256]" = torch.ops.aten.view.default(sum_239, [256]);  sum_239 = None
    permute_513: "f32[256, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_566: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_154, [8, 197, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_680: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_67, 0.5);  add_67 = None
    mul_681: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_682: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_681, -0.5);  mul_681 = None
    exp_27: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_682);  mul_682 = None
    mul_683: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_684: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, mul_683);  view_33 = mul_683 = None
    add_287: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_680, mul_684);  mul_680 = mul_684 = None
    mul_685: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_566, add_287);  view_566 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_567: "f32[1576, 768]" = torch.ops.aten.view.default(mul_685, [1576, 768]);  mul_685 = None
    mm_156: "f32[1576, 256]" = torch.ops.aten.mm.default(view_567, permute_514);  permute_514 = None
    permute_515: "f32[768, 1576]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_157: "f32[768, 256]" = torch.ops.aten.mm.default(permute_515, view_32);  permute_515 = view_32 = None
    permute_516: "f32[256, 768]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_240: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.view.default(sum_240, [768]);  sum_240 = None
    permute_517: "f32[768, 256]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_569: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_156, [8, 197, 256]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_687: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_569, primals_39);  primals_39 = None
    mul_688: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_687, 256)
    sum_241: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True)
    mul_689: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_687, mul_98);  mul_687 = None
    sum_242: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True);  mul_689 = None
    mul_690: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_98, sum_242);  sum_242 = None
    sub_217: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_688, sum_241);  mul_688 = sum_241 = None
    sub_218: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_217, mul_690);  sub_217 = mul_690 = None
    mul_691: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_218);  div_45 = sub_218 = None
    mul_692: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_569, mul_98);  mul_98 = None
    sum_243: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 1]);  mul_692 = None
    sum_244: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_569, [0, 1]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_288: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_285, mul_691);  add_285 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_570: "f32[1576, 256]" = torch.ops.aten.view.default(add_288, [1576, 256])
    mm_158: "f32[1576, 256]" = torch.ops.aten.mm.default(view_570, permute_518);  permute_518 = None
    permute_519: "f32[256, 1576]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_159: "f32[256, 256]" = torch.ops.aten.mm.default(permute_519, view_30);  permute_519 = view_30 = None
    permute_520: "f32[256, 256]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_245: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[256]" = torch.ops.aten.view.default(sum_245, [256]);  sum_245 = None
    permute_521: "f32[256, 256]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    view_572: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_158, [8, 197, 256]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_573: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_572, [8, 197, 4, 64]);  view_572 = None
    permute_522: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_522, getitem_34, getitem_35, getitem_36, alias_33, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44);  permute_522 = getitem_34 = getitem_35 = getitem_36 = alias_33 = getitem_38 = getitem_39 = getitem_40 = getitem_43 = getitem_44 = None
    getitem_259: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[0]
    getitem_260: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[1]
    getitem_261: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_24: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_259, getitem_260, getitem_261]);  getitem_259 = getitem_260 = getitem_261 = None
    view_574: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_24, [3, 8, 4, 197, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_523: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_574, [1, 3, 0, 2, 4]);  view_574 = None
    clone_97: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_575: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_97, [8, 197, 768]);  clone_97 = None
    view_576: "f32[1576, 768]" = torch.ops.aten.view.default(view_575, [1576, 768]);  view_575 = None
    mm_160: "f32[1576, 256]" = torch.ops.aten.mm.default(view_576, permute_524);  permute_524 = None
    permute_525: "f32[768, 1576]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_161: "f32[768, 256]" = torch.ops.aten.mm.default(permute_525, view_26);  permute_525 = view_26 = None
    permute_526: "f32[256, 768]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_246: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[768]" = torch.ops.aten.view.default(sum_246, [768]);  sum_246 = None
    permute_527: "f32[768, 256]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    view_578: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_160, [8, 197, 256]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_694: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_578, primals_33);  primals_33 = None
    mul_695: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_694, 256)
    sum_247: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [2], True)
    mul_696: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_694, mul_96);  mul_694 = None
    sum_248: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True);  mul_696 = None
    mul_697: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_96, sum_248);  sum_248 = None
    sub_220: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_695, sum_247);  mul_695 = sum_247 = None
    sub_221: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_220, mul_697);  sub_220 = mul_697 = None
    mul_698: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_221);  div_46 = sub_221 = None
    mul_699: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_578, mul_96);  mul_96 = None
    sum_249: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 1]);  mul_699 = None
    sum_250: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_578, [0, 1]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_289: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_288, mul_698);  add_288 = mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_579: "f32[1576, 256]" = torch.ops.aten.view.default(add_289, [1576, 256])
    mm_162: "f32[1576, 768]" = torch.ops.aten.mm.default(view_579, permute_528);  permute_528 = None
    permute_529: "f32[256, 1576]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_163: "f32[256, 768]" = torch.ops.aten.mm.default(permute_529, view_24);  permute_529 = view_24 = None
    permute_530: "f32[768, 256]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_251: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[256]" = torch.ops.aten.view.default(sum_251, [256]);  sum_251 = None
    permute_531: "f32[256, 768]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    view_581: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_162, [8, 197, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_701: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_60, 0.5);  add_60 = None
    mul_702: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, view_23)
    mul_703: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_702, -0.5);  mul_702 = None
    exp_28: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_703);  mul_703 = None
    mul_704: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_705: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, mul_704);  view_23 = mul_704 = None
    add_291: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_701, mul_705);  mul_701 = mul_705 = None
    mul_706: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_581, add_291);  view_581 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_582: "f32[1576, 768]" = torch.ops.aten.view.default(mul_706, [1576, 768]);  mul_706 = None
    mm_164: "f32[1576, 256]" = torch.ops.aten.mm.default(view_582, permute_532);  permute_532 = None
    permute_533: "f32[768, 1576]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_165: "f32[768, 256]" = torch.ops.aten.mm.default(permute_533, view_22);  permute_533 = view_22 = None
    permute_534: "f32[256, 768]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_252: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_252, [768]);  sum_252 = None
    permute_535: "f32[768, 256]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    view_584: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_164, [8, 197, 256]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_708: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_584, primals_27);  primals_27 = None
    mul_709: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_708, 256)
    sum_253: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [2], True)
    mul_710: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_708, mul_91);  mul_708 = None
    sum_254: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_710, [2], True);  mul_710 = None
    mul_711: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_91, sum_254);  sum_254 = None
    sub_223: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_709, sum_253);  mul_709 = sum_253 = None
    sub_224: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_223, mul_711);  sub_223 = mul_711 = None
    mul_712: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_47, sub_224);  div_47 = sub_224 = None
    mul_713: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_584, mul_91);  mul_91 = None
    sum_255: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 1]);  mul_713 = None
    sum_256: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_584, [0, 1]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_292: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_289, mul_712);  add_289 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_585: "f32[1576, 256]" = torch.ops.aten.view.default(add_292, [1576, 256])
    mm_166: "f32[1576, 256]" = torch.ops.aten.mm.default(view_585, permute_536);  permute_536 = None
    permute_537: "f32[256, 1576]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_167: "f32[256, 256]" = torch.ops.aten.mm.default(permute_537, view_20);  permute_537 = view_20 = None
    permute_538: "f32[256, 256]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_257: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[256]" = torch.ops.aten.view.default(sum_257, [256]);  sum_257 = None
    permute_539: "f32[256, 256]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_587: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_166, [8, 197, 256]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_588: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_587, [8, 197, 4, 64]);  view_587 = None
    permute_540: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_540, getitem_18, getitem_19, getitem_20, alias_34, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28);  permute_540 = getitem_18 = getitem_19 = getitem_20 = alias_34 = getitem_22 = getitem_23 = getitem_24 = getitem_27 = getitem_28 = None
    getitem_262: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[0]
    getitem_263: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[1]
    getitem_264: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_25: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_262, getitem_263, getitem_264]);  getitem_262 = getitem_263 = getitem_264 = None
    view_589: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_25, [3, 8, 4, 197, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_541: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_589, [1, 3, 0, 2, 4]);  view_589 = None
    clone_98: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
    view_590: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_98, [8, 197, 768]);  clone_98 = None
    view_591: "f32[1576, 768]" = torch.ops.aten.view.default(view_590, [1576, 768]);  view_590 = None
    mm_168: "f32[1576, 256]" = torch.ops.aten.mm.default(view_591, permute_542);  permute_542 = None
    permute_543: "f32[768, 1576]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_169: "f32[768, 256]" = torch.ops.aten.mm.default(permute_543, view_16);  permute_543 = view_16 = None
    permute_544: "f32[256, 768]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_258: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.view.default(sum_258, [768]);  sum_258 = None
    permute_545: "f32[768, 256]" = torch.ops.aten.permute.default(permute_544, [1, 0]);  permute_544 = None
    view_593: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_168, [8, 197, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_715: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_593, primals_21);  primals_21 = None
    mul_716: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_715, 256)
    sum_259: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True)
    mul_717: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_715, mul_89);  mul_715 = None
    sum_260: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True);  mul_717 = None
    mul_718: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_89, sum_260);  sum_260 = None
    sub_226: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_716, sum_259);  mul_716 = sum_259 = None
    sub_227: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_226, mul_718);  sub_226 = mul_718 = None
    mul_719: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_227);  div_48 = sub_227 = None
    mul_720: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_593, mul_89);  mul_89 = None
    sum_261: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 1]);  mul_720 = None
    sum_262: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_593, [0, 1]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_293: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_292, mul_719);  add_292 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_594: "f32[3208, 128]" = torch.ops.aten.view.default(add_281, [3208, 128])
    mm_170: "f32[3208, 384]" = torch.ops.aten.mm.default(view_594, permute_546);  permute_546 = None
    permute_547: "f32[128, 3208]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_171: "f32[128, 384]" = torch.ops.aten.mm.default(permute_547, view_14);  permute_547 = view_14 = None
    permute_548: "f32[384, 128]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_263: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[128]" = torch.ops.aten.view.default(sum_263, [128]);  sum_263 = None
    permute_549: "f32[128, 384]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_596: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_170, [8, 401, 384]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_722: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
    mul_723: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, view_13)
    mul_724: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_723, -0.5);  mul_723 = None
    exp_29: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_724);  mul_724 = None
    mul_725: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_726: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, mul_725);  view_13 = mul_725 = None
    add_295: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_722, mul_726);  mul_722 = mul_726 = None
    mul_727: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_596, add_295);  view_596 = add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_597: "f32[3208, 384]" = torch.ops.aten.view.default(mul_727, [3208, 384]);  mul_727 = None
    mm_172: "f32[3208, 128]" = torch.ops.aten.mm.default(view_597, permute_550);  permute_550 = None
    permute_551: "f32[384, 3208]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_173: "f32[384, 128]" = torch.ops.aten.mm.default(permute_551, view_12);  permute_551 = view_12 = None
    permute_552: "f32[128, 384]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_264: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[384]" = torch.ops.aten.view.default(sum_264, [384]);  sum_264 = None
    permute_553: "f32[384, 128]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_599: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_172, [8, 401, 128]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_729: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_599, primals_15);  primals_15 = None
    mul_730: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_729, 128)
    sum_265: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_729, [2], True)
    mul_731: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_729, mul_84);  mul_729 = None
    sum_266: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True);  mul_731 = None
    mul_732: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_84, sum_266);  sum_266 = None
    sub_229: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_730, sum_265);  mul_730 = sum_265 = None
    sub_230: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_229, mul_732);  sub_229 = mul_732 = None
    mul_733: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_49, sub_230);  div_49 = sub_230 = None
    mul_734: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_599, mul_84);  mul_84 = None
    sum_267: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 1]);  mul_734 = None
    sum_268: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_599, [0, 1]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_296: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_281, mul_733);  add_281 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_600: "f32[3208, 128]" = torch.ops.aten.view.default(add_296, [3208, 128])
    mm_174: "f32[3208, 128]" = torch.ops.aten.mm.default(view_600, permute_554);  permute_554 = None
    permute_555: "f32[128, 3208]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_175: "f32[128, 128]" = torch.ops.aten.mm.default(permute_555, view_10);  permute_555 = view_10 = None
    permute_556: "f32[128, 128]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_269: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[128]" = torch.ops.aten.view.default(sum_269, [128]);  sum_269 = None
    permute_557: "f32[128, 128]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    view_602: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_174, [8, 401, 128]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_603: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_602, [8, 401, 4, 32]);  view_602 = None
    permute_558: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_558, getitem_2, getitem_3, getitem_4, alias_35, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12);  permute_558 = getitem_2 = getitem_3 = getitem_4 = alias_35 = getitem_6 = getitem_7 = getitem_8 = getitem_11 = getitem_12 = None
    getitem_265: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[0]
    getitem_266: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[1]
    getitem_267: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_26: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_265, getitem_266, getitem_267]);  getitem_265 = getitem_266 = getitem_267 = None
    view_604: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_26, [3, 8, 4, 401, 32]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_559: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_604, [1, 3, 0, 2, 4]);  view_604 = None
    clone_99: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_605: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_99, [8, 401, 384]);  clone_99 = None
    view_606: "f32[3208, 384]" = torch.ops.aten.view.default(view_605, [3208, 384]);  view_605 = None
    mm_176: "f32[3208, 128]" = torch.ops.aten.mm.default(view_606, permute_560);  permute_560 = None
    permute_561: "f32[384, 3208]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_177: "f32[384, 128]" = torch.ops.aten.mm.default(permute_561, view_6);  permute_561 = view_6 = None
    permute_562: "f32[128, 384]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_270: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[384]" = torch.ops.aten.view.default(sum_270, [384]);  sum_270 = None
    permute_563: "f32[384, 128]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    view_608: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_176, [8, 401, 128]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_736: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_608, primals_9);  primals_9 = None
    mul_737: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_736, 128)
    sum_271: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_736, [2], True)
    mul_738: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_736, mul_82);  mul_736 = None
    sum_272: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2], True);  mul_738 = None
    mul_739: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_82, sum_272);  sum_272 = None
    sub_232: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_737, sum_271);  mul_737 = sum_271 = None
    sub_233: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_232, mul_739);  sub_232 = mul_739 = None
    mul_740: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_50, sub_233);  div_50 = sub_233 = None
    mul_741: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_608, mul_82);  mul_82 = None
    sum_273: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 1]);  mul_741 = None
    sum_274: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1]);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_297: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_296, mul_740);  add_296 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    sum_275: "f32[1, 197, 256]" = torch.ops.aten.sum.dim_IntList(add_293, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    slice_93: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_293, 1, 0, 1)
    slice_94: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_293, 1, 1, 197);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    sum_276: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(slice_93, [0], True);  slice_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_564: "f32[8, 256, 196]" = torch.ops.aten.permute.default(slice_94, [0, 2, 1]);  slice_94 = None
    view_609: "f32[8, 256, 14, 14]" = torch.ops.aten.view.default(permute_564, [8, 256, 14, 14]);  permute_564 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(view_609, add_46, primals_7, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_609 = add_46 = primals_7 = None
    getitem_269: "f32[256, 3, 16, 16]" = convolution_backward[1]
    getitem_270: "f32[256]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    sum_277: "f32[1, 401, 128]" = torch.ops.aten.sum.dim_IntList(add_297, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    slice_95: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_297, 1, 0, 1)
    slice_96: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_297, 1, 1, 401);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    sum_278: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(slice_95, [0], True);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_565: "f32[8, 128, 400]" = torch.ops.aten.permute.default(slice_96, [0, 2, 1]);  slice_96 = None
    view_610: "f32[8, 128, 20, 20]" = torch.ops.aten.view.default(permute_565, [8, 128, 20, 20]);  permute_565 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_610, primals_269, primals_5, [128], [12, 12], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_610 = primals_269 = primals_5 = None
    getitem_272: "f32[128, 3, 12, 12]" = convolution_backward_1[1]
    getitem_273: "f32[128]" = convolution_backward_1[2];  convolution_backward_1 = None
    return [sum_278, sum_277, sum_276, sum_275, getitem_272, getitem_273, getitem_269, getitem_270, sum_273, sum_274, permute_563, view_607, permute_557, view_601, sum_267, sum_268, permute_553, view_598, permute_549, view_595, sum_261, sum_262, permute_545, view_592, permute_539, view_586, sum_255, sum_256, permute_535, view_583, permute_531, view_580, sum_249, sum_250, permute_527, view_577, permute_521, view_571, sum_243, sum_244, permute_517, view_568, permute_513, view_565, sum_237, sum_238, permute_509, view_562, permute_503, view_556, sum_231, sum_232, permute_499, view_553, permute_495, view_550, sum_225, sum_226, permute_491, view_547, sum_220, sum_221, permute_487, view_544, sum_215, sum_216, permute_483, view_540, permute_478, view_537, permute_473, view_533, permute_462, view_522, sum_206, sum_207, permute_458, view_519, sum_201, sum_202, permute_454, view_515, permute_449, view_512, permute_444, view_508, permute_433, view_497, sum_192, sum_193, permute_429, view_494, sum_187, sum_188, permute_425, view_491, permute_419, view_485, sum_181, sum_182, permute_415, view_482, permute_411, view_479, sum_175, sum_176, permute_407, view_476, permute_401, view_470, sum_169, sum_170, permute_397, view_467, permute_393, view_464, sum_163, sum_164, permute_389, view_461, permute_383, view_455, sum_157, sum_158, permute_379, view_452, permute_375, view_449, sum_151, sum_152, permute_371, view_446, permute_365, view_440, sum_145, sum_146, permute_361, view_437, permute_357, view_434, sum_139, sum_140, permute_353, view_431, sum_134, sum_135, permute_349, view_428, sum_129, sum_130, permute_345, view_424, permute_340, view_421, permute_335, view_417, permute_324, view_406, sum_120, sum_121, permute_320, view_403, sum_115, sum_116, permute_316, view_399, permute_311, view_396, permute_306, view_392, permute_295, view_381, sum_106, sum_107, permute_291, view_378, sum_101, sum_102, permute_287, view_375, permute_281, view_369, sum_95, sum_96, permute_277, view_366, permute_273, view_363, sum_89, sum_90, permute_269, view_360, permute_263, view_354, sum_83, sum_84, permute_259, view_351, permute_255, view_348, sum_77, sum_78, permute_251, view_345, permute_245, view_339, sum_71, sum_72, permute_241, view_336, permute_237, view_333, sum_65, sum_66, permute_233, view_330, permute_227, view_324, sum_59, sum_60, permute_223, view_321, permute_219, view_318, sum_53, sum_54, permute_215, view_315, sum_48, sum_49, permute_211, view_312, sum_43, sum_44, permute_207, view_308, permute_202, view_305, permute_197, view_301, permute_186, view_290, sum_34, sum_35, permute_182, view_287, sum_29, sum_30, permute_178, view_283, permute_173, view_280, permute_168, view_276, permute_157, view_265, sum_20, sum_21, permute_153, view_262, sum_15, sum_16, sum_11, sum_12, permute_149, view_260, permute_145, view_259, None]
    