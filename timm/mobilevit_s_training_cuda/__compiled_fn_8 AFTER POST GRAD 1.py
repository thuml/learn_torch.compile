from __future__ import annotations



def forward(self, primals_1: "f32[16]", primals_3: "f32[64]", primals_5: "f32[64]", primals_7: "f32[32]", primals_9: "f32[128]", primals_11: "f32[128]", primals_13: "f32[64]", primals_15: "f32[256]", primals_17: "f32[256]", primals_19: "f32[64]", primals_21: "f32[256]", primals_23: "f32[256]", primals_25: "f32[64]", primals_27: "f32[256]", primals_29: "f32[256]", primals_31: "f32[96]", primals_33: "f32[96]", primals_35: "f32[96]", primals_37: "f32[96]", primals_39: "f32[384]", primals_41: "f32[384]", primals_43: "f32[128]", primals_45: "f32[128]", primals_47: "f32[128]", primals_49: "f32[128]", primals_51: "f32[512]", primals_53: "f32[512]", primals_55: "f32[160]", primals_57: "f32[160]", primals_59: "f32[160]", primals_61: "f32[160]", primals_63: "f32[640]", primals_65: "f32[16, 3, 3, 3]", primals_66: "f32[64, 16, 1, 1]", primals_67: "f32[64, 1, 3, 3]", primals_68: "f32[32, 64, 1, 1]", primals_69: "f32[128, 32, 1, 1]", primals_70: "f32[128, 1, 3, 3]", primals_71: "f32[64, 128, 1, 1]", primals_72: "f32[256, 64, 1, 1]", primals_73: "f32[256, 1, 3, 3]", primals_74: "f32[64, 256, 1, 1]", primals_75: "f32[256, 64, 1, 1]", primals_76: "f32[256, 1, 3, 3]", primals_77: "f32[64, 256, 1, 1]", primals_78: "f32[256, 64, 1, 1]", primals_79: "f32[256, 1, 3, 3]", primals_80: "f32[96, 256, 1, 1]", primals_81: "f32[96, 96, 3, 3]", primals_82: "f32[144, 96, 1, 1]", primals_83: "f32[144]", primals_89: "f32[144]", primals_95: "f32[144]", primals_101: "f32[144]", primals_107: "f32[144]", primals_109: "f32[96, 144, 1, 1]", primals_110: "f32[96, 192, 3, 3]", primals_111: "f32[384, 96, 1, 1]", primals_112: "f32[384, 1, 3, 3]", primals_113: "f32[128, 384, 1, 1]", primals_114: "f32[128, 128, 3, 3]", primals_115: "f32[192, 128, 1, 1]", primals_116: "f32[192]", primals_122: "f32[192]", primals_128: "f32[192]", primals_134: "f32[192]", primals_140: "f32[192]", primals_146: "f32[192]", primals_152: "f32[192]", primals_158: "f32[192]", primals_164: "f32[192]", primals_166: "f32[128, 192, 1, 1]", primals_167: "f32[128, 256, 3, 3]", primals_168: "f32[512, 128, 1, 1]", primals_169: "f32[512, 1, 3, 3]", primals_170: "f32[160, 512, 1, 1]", primals_171: "f32[160, 160, 3, 3]", primals_172: "f32[240, 160, 1, 1]", primals_173: "f32[240]", primals_179: "f32[240]", primals_185: "f32[240]", primals_191: "f32[240]", primals_197: "f32[240]", primals_203: "f32[240]", primals_209: "f32[240]", primals_211: "f32[160, 240, 1, 1]", primals_212: "f32[160, 320, 3, 3]", primals_213: "f32[640, 160, 1, 1]", primals_312: "f32[8, 3, 256, 256]", convolution: "f32[8, 16, 128, 128]", squeeze_1: "f32[16]", mul_7: "f32[8, 16, 128, 128]", convolution_1: "f32[8, 64, 128, 128]", squeeze_4: "f32[64]", mul_15: "f32[8, 64, 128, 128]", convolution_2: "f32[8, 64, 128, 128]", squeeze_7: "f32[64]", mul_23: "f32[8, 64, 128, 128]", convolution_3: "f32[8, 32, 128, 128]", squeeze_10: "f32[32]", add_19: "f32[8, 32, 128, 128]", convolution_4: "f32[8, 128, 128, 128]", squeeze_13: "f32[128]", mul_38: "f32[8, 128, 128, 128]", convolution_5: "f32[8, 128, 64, 64]", squeeze_16: "f32[128]", mul_46: "f32[8, 128, 64, 64]", convolution_6: "f32[8, 64, 64, 64]", squeeze_19: "f32[64]", add_34: "f32[8, 64, 64, 64]", convolution_7: "f32[8, 256, 64, 64]", squeeze_22: "f32[256]", mul_61: "f32[8, 256, 64, 64]", convolution_8: "f32[8, 256, 64, 64]", squeeze_25: "f32[256]", mul_69: "f32[8, 256, 64, 64]", convolution_9: "f32[8, 64, 64, 64]", squeeze_28: "f32[64]", add_50: "f32[8, 64, 64, 64]", convolution_10: "f32[8, 256, 64, 64]", squeeze_31: "f32[256]", mul_84: "f32[8, 256, 64, 64]", convolution_11: "f32[8, 256, 64, 64]", squeeze_34: "f32[256]", mul_92: "f32[8, 256, 64, 64]", convolution_12: "f32[8, 64, 64, 64]", squeeze_37: "f32[64]", add_66: "f32[8, 64, 64, 64]", convolution_13: "f32[8, 256, 64, 64]", squeeze_40: "f32[256]", mul_107: "f32[8, 256, 64, 64]", convolution_14: "f32[8, 256, 32, 32]", squeeze_43: "f32[256]", mul_115: "f32[8, 256, 32, 32]", convolution_15: "f32[8, 96, 32, 32]", squeeze_46: "f32[96]", add_81: "f32[8, 96, 32, 32]", convolution_16: "f32[8, 96, 32, 32]", squeeze_49: "f32[96]", mul_130: "f32[8, 96, 32, 32]", mul_131: "f32[32, 256, 144]", view_3: "f32[8192, 144]", getitem_36: "f32[32, 4, 256, 36]", getitem_37: "f32[32, 4, 256, 36]", getitem_38: "f32[32, 4, 256, 36]", getitem_40: "f32[32, 4, 256]", getitem_41: "i64[]", getitem_42: "i64[]", view_7: "f32[8192, 144]", mul_133: "f32[32, 256, 144]", view_9: "f32[8192, 144]", addmm_2: "f32[8192, 288]", view_11: "f32[8192, 288]", mul_136: "f32[32, 256, 144]", view_13: "f32[8192, 144]", getitem_47: "f32[32, 4, 256, 36]", getitem_48: "f32[32, 4, 256, 36]", getitem_49: "f32[32, 4, 256, 36]", getitem_51: "f32[32, 4, 256]", getitem_52: "i64[]", getitem_53: "i64[]", view_17: "f32[8192, 144]", mul_138: "f32[32, 256, 144]", view_19: "f32[8192, 144]", addmm_6: "f32[8192, 288]", view_21: "f32[8192, 288]", mul_141: "f32[32, 256, 144]", view_25: "f32[8, 144, 32, 32]", convolution_18: "f32[8, 96, 32, 32]", squeeze_52: "f32[96]", cat: "f32[8, 192, 32, 32]", convolution_19: "f32[8, 96, 32, 32]", squeeze_55: "f32[96]", mul_158: "f32[8, 96, 32, 32]", convolution_20: "f32[8, 384, 32, 32]", squeeze_58: "f32[384]", mul_166: "f32[8, 384, 32, 32]", convolution_21: "f32[8, 384, 16, 16]", squeeze_61: "f32[384]", mul_174: "f32[8, 384, 16, 16]", convolution_22: "f32[8, 128, 16, 16]", squeeze_64: "f32[128]", add_125: "f32[8, 128, 16, 16]", convolution_23: "f32[8, 128, 16, 16]", squeeze_67: "f32[128]", mul_189: "f32[8, 128, 16, 16]", mul_190: "f32[32, 64, 192]", view_29: "f32[2048, 192]", getitem_72: "f32[32, 4, 64, 48]", getitem_73: "f32[32, 4, 64, 48]", getitem_74: "f32[32, 4, 64, 48]", getitem_76: "f32[32, 4, 64]", getitem_77: "i64[]", getitem_78: "i64[]", view_33: "f32[2048, 192]", mul_192: "f32[32, 64, 192]", view_35: "f32[2048, 192]", addmm_10: "f32[2048, 384]", view_37: "f32[2048, 384]", mul_195: "f32[32, 64, 192]", view_39: "f32[2048, 192]", getitem_83: "f32[32, 4, 64, 48]", getitem_84: "f32[32, 4, 64, 48]", getitem_85: "f32[32, 4, 64, 48]", getitem_87: "f32[32, 4, 64]", getitem_88: "i64[]", getitem_89: "i64[]", view_43: "f32[2048, 192]", mul_197: "f32[32, 64, 192]", view_45: "f32[2048, 192]", addmm_14: "f32[2048, 384]", view_47: "f32[2048, 384]", mul_200: "f32[32, 64, 192]", view_49: "f32[2048, 192]", getitem_94: "f32[32, 4, 64, 48]", getitem_95: "f32[32, 4, 64, 48]", getitem_96: "f32[32, 4, 64, 48]", getitem_98: "f32[32, 4, 64]", getitem_99: "i64[]", getitem_100: "i64[]", view_53: "f32[2048, 192]", mul_202: "f32[32, 64, 192]", view_55: "f32[2048, 192]", addmm_18: "f32[2048, 384]", view_57: "f32[2048, 384]", mul_205: "f32[32, 64, 192]", view_59: "f32[2048, 192]", getitem_105: "f32[32, 4, 64, 48]", getitem_106: "f32[32, 4, 64, 48]", getitem_107: "f32[32, 4, 64, 48]", getitem_109: "f32[32, 4, 64]", getitem_110: "i64[]", getitem_111: "i64[]", view_63: "f32[2048, 192]", mul_207: "f32[32, 64, 192]", view_65: "f32[2048, 192]", addmm_22: "f32[2048, 384]", view_67: "f32[2048, 384]", mul_210: "f32[32, 64, 192]", view_71: "f32[8, 192, 16, 16]", convolution_25: "f32[8, 128, 16, 16]", squeeze_70: "f32[128]", cat_1: "f32[8, 256, 16, 16]", convolution_26: "f32[8, 128, 16, 16]", squeeze_73: "f32[128]", mul_227: "f32[8, 128, 16, 16]", convolution_27: "f32[8, 512, 16, 16]", squeeze_76: "f32[512]", mul_235: "f32[8, 512, 16, 16]", convolution_28: "f32[8, 512, 8, 8]", squeeze_79: "f32[512]", mul_243: "f32[8, 512, 8, 8]", convolution_29: "f32[8, 160, 8, 8]", squeeze_82: "f32[160]", add_181: "f32[8, 160, 8, 8]", convolution_30: "f32[8, 160, 8, 8]", squeeze_85: "f32[160]", mul_258: "f32[8, 160, 8, 8]", mul_259: "f32[32, 16, 240]", view_75: "f32[512, 240]", getitem_130: "f32[32, 4, 16, 60]", getitem_131: "f32[32, 4, 16, 60]", getitem_132: "f32[32, 4, 16, 60]", getitem_134: "f32[32, 4, 32]", getitem_135: "i64[]", getitem_136: "i64[]", view_79: "f32[512, 240]", mul_261: "f32[32, 16, 240]", view_81: "f32[512, 240]", addmm_26: "f32[512, 480]", view_83: "f32[512, 480]", mul_264: "f32[32, 16, 240]", view_85: "f32[512, 240]", getitem_141: "f32[32, 4, 16, 60]", getitem_142: "f32[32, 4, 16, 60]", getitem_143: "f32[32, 4, 16, 60]", getitem_145: "f32[32, 4, 32]", getitem_146: "i64[]", getitem_147: "i64[]", view_89: "f32[512, 240]", mul_266: "f32[32, 16, 240]", view_91: "f32[512, 240]", addmm_30: "f32[512, 480]", view_93: "f32[512, 480]", mul_269: "f32[32, 16, 240]", view_95: "f32[512, 240]", getitem_152: "f32[32, 4, 16, 60]", getitem_153: "f32[32, 4, 16, 60]", getitem_154: "f32[32, 4, 16, 60]", getitem_156: "f32[32, 4, 32]", getitem_157: "i64[]", getitem_158: "i64[]", view_99: "f32[512, 240]", mul_271: "f32[32, 16, 240]", view_101: "f32[512, 240]", addmm_34: "f32[512, 480]", view_103: "f32[512, 480]", mul_274: "f32[32, 16, 240]", view_107: "f32[8, 240, 8, 8]", convolution_32: "f32[8, 160, 8, 8]", squeeze_88: "f32[160]", cat_2: "f32[8, 320, 8, 8]", convolution_33: "f32[8, 160, 8, 8]", squeeze_91: "f32[160]", mul_291: "f32[8, 160, 8, 8]", convolution_34: "f32[8, 640, 8, 8]", squeeze_94: "f32[640]", clone_64: "f32[8, 640]", permute_67: "f32[1000, 640]", mul_301: "f32[8, 640, 8, 8]", unsqueeze_130: "f32[1, 640, 1, 1]", mul_313: "f32[8, 160, 8, 8]", unsqueeze_142: "f32[1, 160, 1, 1]", mul_325: "f32[8, 160, 8, 8]", unsqueeze_154: "f32[1, 160, 1, 1]", div_1: "f32[32, 16, 1]", permute_76: "f32[240, 480]", permute_81: "f32[480, 240]", div_2: "f32[32, 16, 1]", permute_85: "f32[240, 240]", alias_9: "f32[32, 4, 16, 60]", permute_91: "f32[720, 240]", div_3: "f32[32, 16, 1]", permute_95: "f32[240, 480]", permute_100: "f32[480, 240]", div_4: "f32[32, 16, 1]", permute_104: "f32[240, 240]", alias_10: "f32[32, 4, 16, 60]", permute_110: "f32[720, 240]", div_5: "f32[32, 16, 1]", permute_114: "f32[240, 480]", permute_119: "f32[480, 240]", div_6: "f32[32, 16, 1]", permute_123: "f32[240, 240]", alias_11: "f32[32, 4, 16, 60]", permute_129: "f32[720, 240]", div_7: "f32[32, 16, 1]", mul_395: "f32[8, 160, 8, 8]", unsqueeze_166: "f32[1, 160, 1, 1]", unsqueeze_178: "f32[1, 160, 1, 1]", mul_416: "f32[8, 512, 8, 8]", unsqueeze_190: "f32[1, 512, 1, 1]", mul_428: "f32[8, 512, 16, 16]", unsqueeze_202: "f32[1, 512, 1, 1]", mul_440: "f32[8, 128, 16, 16]", unsqueeze_214: "f32[1, 128, 1, 1]", mul_452: "f32[8, 128, 16, 16]", unsqueeze_226: "f32[1, 128, 1, 1]", div_8: "f32[32, 64, 1]", permute_142: "f32[192, 384]", permute_147: "f32[384, 192]", div_9: "f32[32, 64, 1]", permute_151: "f32[192, 192]", alias_12: "f32[32, 4, 64, 48]", permute_157: "f32[576, 192]", div_10: "f32[32, 64, 1]", permute_161: "f32[192, 384]", permute_166: "f32[384, 192]", div_11: "f32[32, 64, 1]", permute_170: "f32[192, 192]", alias_13: "f32[32, 4, 64, 48]", permute_176: "f32[576, 192]", div_12: "f32[32, 64, 1]", permute_180: "f32[192, 384]", permute_185: "f32[384, 192]", div_13: "f32[32, 64, 1]", permute_189: "f32[192, 192]", alias_14: "f32[32, 4, 64, 48]", permute_195: "f32[576, 192]", div_14: "f32[32, 64, 1]", permute_199: "f32[192, 384]", permute_204: "f32[384, 192]", div_15: "f32[32, 64, 1]", permute_208: "f32[192, 192]", alias_15: "f32[32, 4, 64, 48]", permute_214: "f32[576, 192]", div_16: "f32[32, 64, 1]", mul_539: "f32[8, 128, 16, 16]", unsqueeze_238: "f32[1, 128, 1, 1]", unsqueeze_250: "f32[1, 128, 1, 1]", mul_560: "f32[8, 384, 16, 16]", unsqueeze_262: "f32[1, 384, 1, 1]", mul_572: "f32[8, 384, 32, 32]", unsqueeze_274: "f32[1, 384, 1, 1]", mul_584: "f32[8, 96, 32, 32]", unsqueeze_286: "f32[1, 96, 1, 1]", mul_596: "f32[8, 96, 32, 32]", unsqueeze_298: "f32[1, 96, 1, 1]", div_17: "f32[32, 256, 1]", permute_227: "f32[144, 288]", permute_232: "f32[288, 144]", div_18: "f32[32, 256, 1]", permute_236: "f32[144, 144]", alias_16: "f32[32, 4, 256, 36]", permute_242: "f32[432, 144]", div_19: "f32[32, 256, 1]", permute_246: "f32[144, 288]", permute_251: "f32[288, 144]", div_20: "f32[32, 256, 1]", permute_255: "f32[144, 144]", alias_17: "f32[32, 4, 256, 36]", permute_261: "f32[432, 144]", div_21: "f32[32, 256, 1]", mul_649: "f32[8, 96, 32, 32]", unsqueeze_310: "f32[1, 96, 1, 1]", unsqueeze_322: "f32[1, 96, 1, 1]", mul_670: "f32[8, 256, 32, 32]", unsqueeze_334: "f32[1, 256, 1, 1]", mul_682: "f32[8, 256, 64, 64]", unsqueeze_346: "f32[1, 256, 1, 1]", unsqueeze_358: "f32[1, 64, 1, 1]", mul_703: "f32[8, 256, 64, 64]", unsqueeze_370: "f32[1, 256, 1, 1]", mul_715: "f32[8, 256, 64, 64]", unsqueeze_382: "f32[1, 256, 1, 1]", unsqueeze_394: "f32[1, 64, 1, 1]", mul_736: "f32[8, 256, 64, 64]", unsqueeze_406: "f32[1, 256, 1, 1]", mul_748: "f32[8, 256, 64, 64]", unsqueeze_418: "f32[1, 256, 1, 1]", unsqueeze_430: "f32[1, 64, 1, 1]", mul_769: "f32[8, 128, 64, 64]", unsqueeze_442: "f32[1, 128, 1, 1]", mul_781: "f32[8, 128, 128, 128]", unsqueeze_454: "f32[1, 128, 1, 1]", unsqueeze_466: "f32[1, 32, 1, 1]", mul_802: "f32[8, 64, 128, 128]", unsqueeze_478: "f32[1, 64, 1, 1]", mul_814: "f32[8, 64, 128, 128]", unsqueeze_490: "f32[1, 64, 1, 1]", mul_826: "f32[8, 16, 128, 128]", unsqueeze_502: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(addmm_2, [32, 256, 288]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_12: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_20: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(addmm_6, [32, 256, 288]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_13: "f32[32, 256, 288]" = torch.ops.aten.sigmoid.default(view_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_36: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_10, [32, 64, 384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_19: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_14, [32, 64, 384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_20: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_56: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_18, [32, 64, 384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_21: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(addmm_22, [32, 64, 384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_22: "f32[32, 64, 384]" = torch.ops.aten.sigmoid.default(view_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_82: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_26, [32, 16, 480]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_28: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_92: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_30, [32, 16, 480]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_29: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(addmm_34, [32, 16, 480]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sigmoid_30: "f32[32, 16, 480]" = torch.ops.aten.sigmoid.default(view_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 640]" = torch.ops.aten.mm.default(tangents_1, permute_67);  permute_67 = None
    permute_68: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 640]" = torch.ops.aten.mm.default(permute_68, clone_64);  permute_68 = clone_64 = None
    permute_69: "f32[640, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_109: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_70: "f32[1000, 640]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_110: "f32[8, 640, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 640, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 640, 8, 8]" = torch.ops.aten.expand.default(view_110, [8, 640, 8, 8]);  view_110 = None
    div: "f32[8, 640, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_302: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(div, mul_301);  div = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3])
    sub_54: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
    mul_303: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(mul_302, sub_54)
    sum_3: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 2, 3]);  mul_303 = None
    mul_304: "f32[640]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_131: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
    unsqueeze_132: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
    unsqueeze_133: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
    mul_305: "f32[640]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_306: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_307: "f32[640]" = torch.ops.aten.mul.Tensor(mul_305, mul_306);  mul_305 = mul_306 = None
    unsqueeze_134: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
    unsqueeze_135: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
    unsqueeze_136: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
    mul_308: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_137: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_308, 0);  mul_308 = None
    unsqueeze_138: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
    unsqueeze_139: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
    mul_309: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_136);  sub_54 = unsqueeze_136 = None
    sub_56: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(mul_302, mul_309);  mul_302 = mul_309 = None
    sub_57: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_133);  sub_56 = unsqueeze_133 = None
    mul_310: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_139);  sub_57 = unsqueeze_139 = None
    mul_311: "f32[640]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_94);  sum_3 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_310, mul_291, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_310 = mul_291 = primals_213 = None
    getitem_169: "f32[8, 160, 8, 8]" = convolution_backward[0]
    getitem_170: "f32[640, 160, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_314: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_169, mul_313);  getitem_169 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3])
    sub_59: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_142);  convolution_33 = unsqueeze_142 = None
    mul_315: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_314, sub_59)
    sum_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 2, 3]);  mul_315 = None
    mul_316: "f32[160]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_143: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
    unsqueeze_144: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
    unsqueeze_145: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
    mul_317: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_318: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_319: "f32[160]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
    unsqueeze_146: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_147: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
    unsqueeze_148: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
    mul_320: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_149: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_150: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
    unsqueeze_151: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
    mul_321: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_148);  sub_59 = unsqueeze_148 = None
    sub_61: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_314, mul_321);  mul_314 = mul_321 = None
    sub_62: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_145);  sub_61 = unsqueeze_145 = None
    mul_322: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_151);  sub_62 = unsqueeze_151 = None
    mul_323: "f32[160]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_91);  sum_5 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_322, cat_2, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_322 = cat_2 = primals_212 = None
    getitem_172: "f32[8, 320, 8, 8]" = convolution_backward_1[0]
    getitem_173: "f32[160, 320, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_1: "f32[8, 160, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_172, 1, 0, 160)
    slice_2: "f32[8, 160, 8, 8]" = torch.ops.aten.slice.Tensor(getitem_172, 1, 160, 320);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_326: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(slice_2, mul_325);  slice_2 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3])
    sub_64: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_154);  convolution_32 = unsqueeze_154 = None
    mul_327: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_326, sub_64)
    sum_7: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
    mul_328: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_155: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_156: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
    unsqueeze_157: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
    mul_329: "f32[160]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_330: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_331: "f32[160]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
    unsqueeze_158: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_159: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
    unsqueeze_160: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
    mul_332: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_161: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_162: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
    unsqueeze_163: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
    mul_333: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_160);  sub_64 = unsqueeze_160 = None
    sub_66: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_326, mul_333);  mul_326 = mul_333 = None
    sub_67: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_157);  sub_66 = unsqueeze_157 = None
    mul_334: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_163);  sub_67 = unsqueeze_163 = None
    mul_335: "f32[160]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_88);  sum_7 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_334, view_107, primals_211, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = view_107 = primals_211 = None
    getitem_175: "f32[8, 240, 8, 8]" = convolution_backward_2[0]
    getitem_176: "f32[160, 240, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_111: "f32[7680, 2, 4, 2]" = torch.ops.aten.reshape.default(getitem_175, [7680, 2, 4, 2]);  getitem_175 = None
    permute_74: "f32[7680, 4, 2, 2]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_65: "f32[7680, 4, 2, 2]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    view_112: "f32[8, 240, 16, 4]" = torch.ops.aten.reshape.default(clone_65, [8, 240, 16, 4]);  clone_65 = None
    permute_75: "f32[8, 4, 16, 240]" = torch.ops.aten.permute.default(view_112, [0, 3, 2, 1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_66: "f32[8, 4, 16, 240]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    view_113: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(clone_66, [32, 16, 240]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    mul_337: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_113, primals_209);  primals_209 = None
    mul_338: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_337, 240)
    sum_8: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
    mul_339: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_337, mul_274);  mul_337 = None
    sum_9: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
    mul_340: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_274, sum_9);  sum_9 = None
    sub_69: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_338, sum_8);  mul_338 = sum_8 = None
    sub_70: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_69, mul_340);  sub_69 = mul_340 = None
    mul_341: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_1, sub_70);  div_1 = sub_70 = None
    mul_342: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_113, mul_274);  mul_274 = None
    sum_10: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
    sum_11: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_114: "f32[512, 240]" = torch.ops.aten.reshape.default(mul_341, [512, 240])
    mm_2: "f32[512, 480]" = torch.ops.aten.mm.default(view_114, permute_76);  permute_76 = None
    permute_77: "f32[240, 512]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_3: "f32[240, 480]" = torch.ops.aten.mm.default(permute_77, view_103);  permute_77 = view_103 = None
    permute_78: "f32[480, 240]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_12: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
    view_115: "f32[240]" = torch.ops.aten.reshape.default(sum_12, [240]);  sum_12 = None
    permute_79: "f32[240, 480]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    view_116: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(mm_2, [32, 16, 480]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    full_default_3: "f32[32, 16, 480]" = torch.ops.aten.full.default([32, 16, 480], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_71: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_default_3, sigmoid_30)
    mul_343: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_102, sub_71);  view_102 = sub_71 = None
    add_225: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_343, 1);  mul_343 = None
    mul_344: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_30, add_225);  sigmoid_30 = add_225 = None
    mul_345: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_116, mul_344);  view_116 = mul_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_345, [512, 480]);  mul_345 = None
    mm_4: "f32[512, 240]" = torch.ops.aten.mm.default(view_117, permute_81);  permute_81 = None
    permute_82: "f32[480, 512]" = torch.ops.aten.permute.default(view_117, [1, 0])
    mm_5: "f32[480, 240]" = torch.ops.aten.mm.default(permute_82, view_101);  permute_82 = view_101 = None
    permute_83: "f32[240, 480]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_13: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
    view_118: "f32[480]" = torch.ops.aten.reshape.default(sum_13, [480]);  sum_13 = None
    permute_84: "f32[480, 240]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    view_119: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_4, [32, 16, 240]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_347: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_119, primals_203);  primals_203 = None
    mul_348: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_347, 240)
    sum_14: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_347, mul_271);  mul_347 = None
    sum_15: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_271, sum_15);  sum_15 = None
    sub_73: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_348, sum_14);  mul_348 = sum_14 = None
    sub_74: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_73, mul_350);  sub_73 = mul_350 = None
    mul_351: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_2, sub_74);  div_2 = sub_74 = None
    mul_352: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_119, mul_271);  mul_271 = None
    sum_16: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_17: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_119, [0, 1]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_226: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(mul_341, mul_351);  mul_341 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_120: "f32[512, 240]" = torch.ops.aten.reshape.default(add_226, [512, 240])
    mm_6: "f32[512, 240]" = torch.ops.aten.mm.default(view_120, permute_85);  permute_85 = None
    permute_86: "f32[240, 512]" = torch.ops.aten.permute.default(view_120, [1, 0])
    mm_7: "f32[240, 240]" = torch.ops.aten.mm.default(permute_86, view_99);  permute_86 = view_99 = None
    permute_87: "f32[240, 240]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_18: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
    view_121: "f32[240]" = torch.ops.aten.reshape.default(sum_18, [240]);  sum_18 = None
    permute_88: "f32[240, 240]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    view_122: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_6, [32, 16, 240]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_123: "f32[32, 16, 4, 60]" = torch.ops.aten.reshape.default(view_122, [32, 16, 4, 60]);  view_122 = None
    permute_89: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_89, getitem_152, getitem_153, getitem_154, None, alias_9, getitem_156, getitem_157, getitem_158, 0.0, [True, True, True, False]);  permute_89 = getitem_152 = getitem_153 = getitem_154 = alias_9 = getitem_156 = getitem_157 = getitem_158 = None
    getitem_178: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_179: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_180: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
    view_124: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.reshape.default(cat_3, [3, 32, 4, 16, 60]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_90: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_124, [1, 3, 0, 2, 4]);  view_124 = None
    clone_67: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    view_125: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(clone_67, [32, 16, 720]);  clone_67 = None
    view_126: "f32[512, 720]" = torch.ops.aten.reshape.default(view_125, [512, 720]);  view_125 = None
    mm_8: "f32[512, 240]" = torch.ops.aten.mm.default(view_126, permute_91);  permute_91 = None
    permute_92: "f32[720, 512]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_9: "f32[720, 240]" = torch.ops.aten.mm.default(permute_92, view_95);  permute_92 = view_95 = None
    permute_93: "f32[240, 720]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_19: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[720]" = torch.ops.aten.reshape.default(sum_19, [720]);  sum_19 = None
    permute_94: "f32[720, 240]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    view_128: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_8, [32, 16, 240]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_354: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_128, primals_197);  primals_197 = None
    mul_355: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_354, 240)
    sum_20: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_354, mul_269);  mul_354 = None
    sum_21: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_269, sum_21);  sum_21 = None
    sub_76: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_355, sum_20);  mul_355 = sum_20 = None
    sub_77: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_76, mul_357);  sub_76 = mul_357 = None
    mul_358: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_3, sub_77);  div_3 = sub_77 = None
    mul_359: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_128, mul_269);  mul_269 = None
    sum_22: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_23: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_227: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_226, mul_358);  add_226 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[512, 240]" = torch.ops.aten.reshape.default(add_227, [512, 240])
    mm_10: "f32[512, 480]" = torch.ops.aten.mm.default(view_129, permute_95);  permute_95 = None
    permute_96: "f32[240, 512]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_11: "f32[240, 480]" = torch.ops.aten.mm.default(permute_96, view_93);  permute_96 = view_93 = None
    permute_97: "f32[480, 240]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_24: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[240]" = torch.ops.aten.reshape.default(sum_24, [240]);  sum_24 = None
    permute_98: "f32[240, 480]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    view_131: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(mm_10, [32, 16, 480]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_78: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_default_3, sigmoid_29)
    mul_360: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_92, sub_78);  view_92 = sub_78 = None
    add_228: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_360, 1);  mul_360 = None
    mul_361: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_29, add_228);  sigmoid_29 = add_228 = None
    mul_362: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_131, mul_361);  view_131 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_132: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_362, [512, 480]);  mul_362 = None
    mm_12: "f32[512, 240]" = torch.ops.aten.mm.default(view_132, permute_100);  permute_100 = None
    permute_101: "f32[480, 512]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_13: "f32[480, 240]" = torch.ops.aten.mm.default(permute_101, view_91);  permute_101 = view_91 = None
    permute_102: "f32[240, 480]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_25: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
    view_133: "f32[480]" = torch.ops.aten.reshape.default(sum_25, [480]);  sum_25 = None
    permute_103: "f32[480, 240]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_134: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_12, [32, 16, 240]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_364: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_134, primals_191);  primals_191 = None
    mul_365: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_364, 240)
    sum_26: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
    mul_366: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_364, mul_266);  mul_364 = None
    sum_27: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
    mul_367: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_266, sum_27);  sum_27 = None
    sub_80: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_365, sum_26);  mul_365 = sum_26 = None
    sub_81: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_80, mul_367);  sub_80 = mul_367 = None
    mul_368: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_4, sub_81);  div_4 = sub_81 = None
    mul_369: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_134, mul_266);  mul_266 = None
    sum_28: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
    sum_29: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_134, [0, 1]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_229: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_227, mul_368);  add_227 = mul_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_135: "f32[512, 240]" = torch.ops.aten.reshape.default(add_229, [512, 240])
    mm_14: "f32[512, 240]" = torch.ops.aten.mm.default(view_135, permute_104);  permute_104 = None
    permute_105: "f32[240, 512]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_15: "f32[240, 240]" = torch.ops.aten.mm.default(permute_105, view_89);  permute_105 = view_89 = None
    permute_106: "f32[240, 240]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_30: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[240]" = torch.ops.aten.reshape.default(sum_30, [240]);  sum_30 = None
    permute_107: "f32[240, 240]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_137: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_14, [32, 16, 240]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_138: "f32[32, 16, 4, 60]" = torch.ops.aten.reshape.default(view_137, [32, 16, 4, 60]);  view_137 = None
    permute_108: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_108, getitem_141, getitem_142, getitem_143, None, alias_10, getitem_145, getitem_146, getitem_147, 0.0, [True, True, True, False]);  permute_108 = getitem_141 = getitem_142 = getitem_143 = alias_10 = getitem_145 = getitem_146 = getitem_147 = None
    getitem_182: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_183: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_184: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_182, getitem_183, getitem_184]);  getitem_182 = getitem_183 = getitem_184 = None
    view_139: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.reshape.default(cat_4, [3, 32, 4, 16, 60]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_109: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_139, [1, 3, 0, 2, 4]);  view_139 = None
    clone_68: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_140: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(clone_68, [32, 16, 720]);  clone_68 = None
    view_141: "f32[512, 720]" = torch.ops.aten.reshape.default(view_140, [512, 720]);  view_140 = None
    mm_16: "f32[512, 240]" = torch.ops.aten.mm.default(view_141, permute_110);  permute_110 = None
    permute_111: "f32[720, 512]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_17: "f32[720, 240]" = torch.ops.aten.mm.default(permute_111, view_85);  permute_111 = view_85 = None
    permute_112: "f32[240, 720]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_31: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[720]" = torch.ops.aten.reshape.default(sum_31, [720]);  sum_31 = None
    permute_113: "f32[720, 240]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_143: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_16, [32, 16, 240]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_371: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_143, primals_185);  primals_185 = None
    mul_372: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_371, 240)
    sum_32: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_371, mul_264);  mul_371 = None
    sum_33: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_264, sum_33);  sum_33 = None
    sub_83: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_372, sum_32);  mul_372 = sum_32 = None
    sub_84: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_83, mul_374);  sub_83 = mul_374 = None
    mul_375: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_5, sub_84);  div_5 = sub_84 = None
    mul_376: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_143, mul_264);  mul_264 = None
    sum_34: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_35: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_230: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_229, mul_375);  add_229 = mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_144: "f32[512, 240]" = torch.ops.aten.reshape.default(add_230, [512, 240])
    mm_18: "f32[512, 480]" = torch.ops.aten.mm.default(view_144, permute_114);  permute_114 = None
    permute_115: "f32[240, 512]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_19: "f32[240, 480]" = torch.ops.aten.mm.default(permute_115, view_83);  permute_115 = view_83 = None
    permute_116: "f32[480, 240]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_36: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[240]" = torch.ops.aten.reshape.default(sum_36, [240]);  sum_36 = None
    permute_117: "f32[240, 480]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    view_146: "f32[32, 16, 480]" = torch.ops.aten.reshape.default(mm_18, [32, 16, 480]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_85: "f32[32, 16, 480]" = torch.ops.aten.sub.Tensor(full_default_3, sigmoid_28);  full_default_3 = None
    mul_377: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_82, sub_85);  view_82 = sub_85 = None
    add_231: "f32[32, 16, 480]" = torch.ops.aten.add.Scalar(mul_377, 1);  mul_377 = None
    mul_378: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(sigmoid_28, add_231);  sigmoid_28 = add_231 = None
    mul_379: "f32[32, 16, 480]" = torch.ops.aten.mul.Tensor(view_146, mul_378);  view_146 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_147: "f32[512, 480]" = torch.ops.aten.reshape.default(mul_379, [512, 480]);  mul_379 = None
    mm_20: "f32[512, 240]" = torch.ops.aten.mm.default(view_147, permute_119);  permute_119 = None
    permute_120: "f32[480, 512]" = torch.ops.aten.permute.default(view_147, [1, 0])
    mm_21: "f32[480, 240]" = torch.ops.aten.mm.default(permute_120, view_81);  permute_120 = view_81 = None
    permute_121: "f32[240, 480]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_37: "f32[1, 480]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
    view_148: "f32[480]" = torch.ops.aten.reshape.default(sum_37, [480]);  sum_37 = None
    permute_122: "f32[480, 240]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    view_149: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_20, [32, 16, 240]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_381: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_149, primals_179);  primals_179 = None
    mul_382: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_381, 240)
    sum_38: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
    mul_383: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_381, mul_261);  mul_381 = None
    sum_39: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
    mul_384: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_261, sum_39);  sum_39 = None
    sub_87: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_382, sum_38);  mul_382 = sum_38 = None
    sub_88: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_87, mul_384);  sub_87 = mul_384 = None
    mul_385: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_6, sub_88);  div_6 = sub_88 = None
    mul_386: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_149, mul_261);  mul_261 = None
    sum_40: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
    sum_41: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_149, [0, 1]);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_232: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_230, mul_385);  add_230 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_150: "f32[512, 240]" = torch.ops.aten.reshape.default(add_232, [512, 240])
    mm_22: "f32[512, 240]" = torch.ops.aten.mm.default(view_150, permute_123);  permute_123 = None
    permute_124: "f32[240, 512]" = torch.ops.aten.permute.default(view_150, [1, 0])
    mm_23: "f32[240, 240]" = torch.ops.aten.mm.default(permute_124, view_79);  permute_124 = view_79 = None
    permute_125: "f32[240, 240]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_42: "f32[1, 240]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
    view_151: "f32[240]" = torch.ops.aten.reshape.default(sum_42, [240]);  sum_42 = None
    permute_126: "f32[240, 240]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_152: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_22, [32, 16, 240]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_153: "f32[32, 16, 4, 60]" = torch.ops.aten.reshape.default(view_152, [32, 16, 4, 60]);  view_152 = None
    permute_127: "f32[32, 4, 16, 60]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_127, getitem_130, getitem_131, getitem_132, None, alias_11, getitem_134, getitem_135, getitem_136, 0.0, [True, True, True, False]);  permute_127 = getitem_130 = getitem_131 = getitem_132 = alias_11 = getitem_134 = getitem_135 = getitem_136 = None
    getitem_186: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_187: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_188: "f32[32, 4, 16, 60]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[96, 4, 16, 60]" = torch.ops.aten.cat.default([getitem_186, getitem_187, getitem_188]);  getitem_186 = getitem_187 = getitem_188 = None
    view_154: "f32[3, 32, 4, 16, 60]" = torch.ops.aten.reshape.default(cat_5, [3, 32, 4, 16, 60]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_128: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.permute.default(view_154, [1, 3, 0, 2, 4]);  view_154 = None
    clone_69: "f32[32, 16, 3, 4, 60]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_155: "f32[32, 16, 720]" = torch.ops.aten.reshape.default(clone_69, [32, 16, 720]);  clone_69 = None
    view_156: "f32[512, 720]" = torch.ops.aten.reshape.default(view_155, [512, 720]);  view_155 = None
    mm_24: "f32[512, 240]" = torch.ops.aten.mm.default(view_156, permute_129);  permute_129 = None
    permute_130: "f32[720, 512]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_25: "f32[720, 240]" = torch.ops.aten.mm.default(permute_130, view_75);  permute_130 = view_75 = None
    permute_131: "f32[240, 720]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_43: "f32[1, 720]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
    view_157: "f32[720]" = torch.ops.aten.reshape.default(sum_43, [720]);  sum_43 = None
    permute_132: "f32[720, 240]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    view_158: "f32[32, 16, 240]" = torch.ops.aten.reshape.default(mm_24, [32, 16, 240]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_388: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_158, primals_173);  primals_173 = None
    mul_389: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_388, 240)
    sum_44: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_388, mul_259);  mul_388 = None
    sum_45: "f32[32, 16, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(mul_259, sum_45);  sum_45 = None
    sub_90: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(mul_389, sum_44);  mul_389 = sum_44 = None
    sub_91: "f32[32, 16, 240]" = torch.ops.aten.sub.Tensor(sub_90, mul_391);  sub_90 = mul_391 = None
    mul_392: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(div_7, sub_91);  div_7 = sub_91 = None
    mul_393: "f32[32, 16, 240]" = torch.ops.aten.mul.Tensor(view_158, mul_259);  mul_259 = None
    sum_46: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_47: "f32[240]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_233: "f32[32, 16, 240]" = torch.ops.aten.add.Tensor(add_232, mul_392);  add_232 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_159: "f32[8, 4, 16, 240]" = torch.ops.aten.reshape.default(add_233, [8, 4, 16, 240]);  add_233 = None
    permute_133: "f32[8, 240, 16, 4]" = torch.ops.aten.permute.default(view_159, [0, 3, 2, 1]);  view_159 = None
    clone_70: "f32[8, 240, 16, 4]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_160: "f32[7680, 4, 2, 2]" = torch.ops.aten.reshape.default(clone_70, [7680, 4, 2, 2]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_134: "f32[7680, 2, 4, 2]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
    clone_71: "f32[7680, 2, 4, 2]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    view_161: "f32[8, 240, 8, 8]" = torch.ops.aten.reshape.default(clone_71, [8, 240, 8, 8]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_161, mul_258, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_161 = mul_258 = primals_172 = None
    getitem_190: "f32[8, 160, 8, 8]" = convolution_backward_3[0]
    getitem_191: "f32[240, 160, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_396: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_190, mul_395);  getitem_190 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3])
    sub_93: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_166);  convolution_30 = unsqueeze_166 = None
    mul_397: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(mul_396, sub_93)
    sum_49: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_397, [0, 2, 3]);  mul_397 = None
    mul_398: "f32[160]" = torch.ops.aten.mul.Tensor(sum_48, 0.001953125)
    unsqueeze_167: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
    unsqueeze_168: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_399: "f32[160]" = torch.ops.aten.mul.Tensor(sum_49, 0.001953125)
    mul_400: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_401: "f32[160]" = torch.ops.aten.mul.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
    unsqueeze_170: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_171: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_402: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_173: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
    unsqueeze_174: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    mul_403: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_172);  sub_93 = unsqueeze_172 = None
    sub_95: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(mul_396, mul_403);  mul_396 = mul_403 = None
    sub_96: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_169);  sub_95 = unsqueeze_169 = None
    mul_404: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_175);  sub_96 = unsqueeze_175 = None
    mul_405: "f32[160]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_404, add_181, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_404 = add_181 = primals_171 = None
    getitem_193: "f32[8, 160, 8, 8]" = convolution_backward_4[0]
    getitem_194: "f32[160, 160, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_235: "f32[8, 160, 8, 8]" = torch.ops.aten.add.Tensor(slice_1, getitem_193);  slice_1 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_235, [0, 2, 3])
    sub_97: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_178);  convolution_29 = unsqueeze_178 = None
    mul_406: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(add_235, sub_97)
    sum_51: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
    mul_407: "f32[160]" = torch.ops.aten.mul.Tensor(sum_50, 0.001953125)
    unsqueeze_179: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
    unsqueeze_180: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_408: "f32[160]" = torch.ops.aten.mul.Tensor(sum_51, 0.001953125)
    mul_409: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_410: "f32[160]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    unsqueeze_182: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_183: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_411: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_185: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_186: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    mul_412: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_184);  sub_97 = unsqueeze_184 = None
    sub_99: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(add_235, mul_412);  add_235 = mul_412 = None
    sub_100: "f32[8, 160, 8, 8]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_181);  sub_99 = unsqueeze_181 = None
    mul_413: "f32[8, 160, 8, 8]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_187);  sub_100 = unsqueeze_187 = None
    mul_414: "f32[160]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_413, mul_243, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_413 = mul_243 = primals_170 = None
    getitem_196: "f32[8, 512, 8, 8]" = convolution_backward_5[0]
    getitem_197: "f32[160, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_417: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(getitem_196, mul_416);  getitem_196 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
    sub_102: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_190);  convolution_28 = unsqueeze_190 = None
    mul_418: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(mul_417, sub_102)
    sum_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[512]" = torch.ops.aten.mul.Tensor(sum_52, 0.001953125)
    unsqueeze_191: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_192: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_420: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, 0.001953125)
    mul_421: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_422: "f32[512]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_194: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_195: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_423: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_197: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_198: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    mul_424: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_196);  sub_102 = unsqueeze_196 = None
    sub_104: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
    sub_105: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_193);  sub_104 = unsqueeze_193 = None
    mul_425: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_199);  sub_105 = unsqueeze_199 = None
    mul_426: "f32[512]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_425, mul_235, primals_169, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False]);  mul_425 = mul_235 = primals_169 = None
    getitem_199: "f32[8, 512, 16, 16]" = convolution_backward_6[0]
    getitem_200: "f32[512, 1, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_429: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_199, mul_428);  getitem_199 = mul_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3])
    sub_107: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_202);  convolution_27 = unsqueeze_202 = None
    mul_430: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(mul_429, sub_107)
    sum_55: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3]);  mul_430 = None
    mul_431: "f32[512]" = torch.ops.aten.mul.Tensor(sum_54, 0.00048828125)
    unsqueeze_203: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_204: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_432: "f32[512]" = torch.ops.aten.mul.Tensor(sum_55, 0.00048828125)
    mul_433: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_434: "f32[512]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
    unsqueeze_206: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
    unsqueeze_207: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_209: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_210: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    mul_436: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_208);  sub_107 = unsqueeze_208 = None
    sub_109: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(mul_429, mul_436);  mul_429 = mul_436 = None
    sub_110: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_205);  sub_109 = unsqueeze_205 = None
    mul_437: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_211);  sub_110 = unsqueeze_211 = None
    mul_438: "f32[512]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_437, mul_227, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_437 = mul_227 = primals_168 = None
    getitem_202: "f32[8, 128, 16, 16]" = convolution_backward_7[0]
    getitem_203: "f32[512, 128, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_441: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_202, mul_440);  getitem_202 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_441, [0, 2, 3])
    sub_112: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_214);  convolution_26 = unsqueeze_214 = None
    mul_442: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_441, sub_112)
    sum_57: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 2, 3]);  mul_442 = None
    mul_443: "f32[128]" = torch.ops.aten.mul.Tensor(sum_56, 0.00048828125)
    unsqueeze_215: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
    unsqueeze_216: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_444: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, 0.00048828125)
    mul_445: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_446: "f32[128]" = torch.ops.aten.mul.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    unsqueeze_218: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_219: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_447: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_221: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
    unsqueeze_222: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    mul_448: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_220);  sub_112 = unsqueeze_220 = None
    sub_114: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_441, mul_448);  mul_441 = mul_448 = None
    sub_115: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_217);  sub_114 = unsqueeze_217 = None
    mul_449: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_223);  sub_115 = unsqueeze_223 = None
    mul_450: "f32[128]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_449, cat_1, primals_167, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_449 = cat_1 = primals_167 = None
    getitem_205: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_206: "f32[128, 256, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_3: "f32[8, 128, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_205, 1, 0, 128)
    slice_4: "f32[8, 128, 16, 16]" = torch.ops.aten.slice.Tensor(getitem_205, 1, 128, 256);  getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_453: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(slice_4, mul_452);  slice_4 = mul_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3])
    sub_117: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_226);  convolution_25 = unsqueeze_226 = None
    mul_454: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_453, sub_117)
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_455: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
    unsqueeze_227: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_228: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_456: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
    mul_457: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_458: "f32[128]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_230: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_231: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_459: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_233: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_234: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    mul_460: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_232);  sub_117 = unsqueeze_232 = None
    sub_119: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_453, mul_460);  mul_453 = mul_460 = None
    sub_120: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_229);  sub_119 = unsqueeze_229 = None
    mul_461: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_235);  sub_120 = unsqueeze_235 = None
    mul_462: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_461, view_71, primals_166, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = view_71 = primals_166 = None
    getitem_208: "f32[8, 192, 16, 16]" = convolution_backward_9[0]
    getitem_209: "f32[128, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_162: "f32[12288, 2, 8, 2]" = torch.ops.aten.reshape.default(getitem_208, [12288, 2, 8, 2]);  getitem_208 = None
    permute_140: "f32[12288, 8, 2, 2]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_72: "f32[12288, 8, 2, 2]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_163: "f32[8, 192, 64, 4]" = torch.ops.aten.reshape.default(clone_72, [8, 192, 64, 4]);  clone_72 = None
    permute_141: "f32[8, 4, 64, 192]" = torch.ops.aten.permute.default(view_163, [0, 3, 2, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_73: "f32[8, 4, 64, 192]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    view_164: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(clone_73, [32, 64, 192]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    mul_464: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_164, primals_164);  primals_164 = None
    mul_465: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_464, 192)
    sum_60: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
    mul_466: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_464, mul_210);  mul_464 = None
    sum_61: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
    mul_467: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_210, sum_61);  sum_61 = None
    sub_122: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_465, sum_60);  mul_465 = sum_60 = None
    sub_123: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_122, mul_467);  sub_122 = mul_467 = None
    mul_468: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_8, sub_123);  div_8 = sub_123 = None
    mul_469: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_164, mul_210);  mul_210 = None
    sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
    sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[2048, 192]" = torch.ops.aten.reshape.default(mul_468, [2048, 192])
    mm_26: "f32[2048, 384]" = torch.ops.aten.mm.default(view_165, permute_142);  permute_142 = None
    permute_143: "f32[192, 2048]" = torch.ops.aten.permute.default(view_165, [1, 0])
    mm_27: "f32[192, 384]" = torch.ops.aten.mm.default(permute_143, view_67);  permute_143 = view_67 = None
    permute_144: "f32[384, 192]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_64: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
    view_166: "f32[192]" = torch.ops.aten.reshape.default(sum_64, [192]);  sum_64 = None
    permute_145: "f32[192, 384]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_167: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(mm_26, [32, 64, 384]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    full_default_11: "f32[32, 64, 384]" = torch.ops.aten.full.default([32, 64, 384], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_124: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_default_11, sigmoid_22)
    mul_470: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_66, sub_124);  view_66 = sub_124 = None
    add_240: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_470, 1);  mul_470 = None
    mul_471: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_22, add_240);  sigmoid_22 = add_240 = None
    mul_472: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_167, mul_471);  view_167 = mul_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_168: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_472, [2048, 384]);  mul_472 = None
    mm_28: "f32[2048, 192]" = torch.ops.aten.mm.default(view_168, permute_147);  permute_147 = None
    permute_148: "f32[384, 2048]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_29: "f32[384, 192]" = torch.ops.aten.mm.default(permute_148, view_65);  permute_148 = view_65 = None
    permute_149: "f32[192, 384]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_65: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[384]" = torch.ops.aten.reshape.default(sum_65, [384]);  sum_65 = None
    permute_150: "f32[384, 192]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_170: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_28, [32, 64, 192]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_474: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_170, primals_158);  primals_158 = None
    mul_475: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_474, 192)
    sum_66: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_474, [2], True)
    mul_476: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_474, mul_207);  mul_474 = None
    sum_67: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [2], True);  mul_476 = None
    mul_477: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_207, sum_67);  sum_67 = None
    sub_126: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_475, sum_66);  mul_475 = sum_66 = None
    sub_127: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_126, mul_477);  sub_126 = mul_477 = None
    mul_478: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_9, sub_127);  div_9 = sub_127 = None
    mul_479: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_170, mul_207);  mul_207 = None
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1]);  mul_479 = None
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_241: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(mul_468, mul_478);  mul_468 = mul_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_171: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_241, [2048, 192])
    mm_30: "f32[2048, 192]" = torch.ops.aten.mm.default(view_171, permute_151);  permute_151 = None
    permute_152: "f32[192, 2048]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_31: "f32[192, 192]" = torch.ops.aten.mm.default(permute_152, view_63);  permute_152 = view_63 = None
    permute_153: "f32[192, 192]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_70: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[192]" = torch.ops.aten.reshape.default(sum_70, [192]);  sum_70 = None
    permute_154: "f32[192, 192]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_173: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_30, [32, 64, 192]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_174: "f32[32, 64, 4, 48]" = torch.ops.aten.reshape.default(view_173, [32, 64, 4, 48]);  view_173 = None
    permute_155: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_155, getitem_105, getitem_106, getitem_107, None, alias_12, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False]);  permute_155 = getitem_105 = getitem_106 = getitem_107 = alias_12 = getitem_109 = getitem_110 = getitem_111 = None
    getitem_211: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_212: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_213: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_211, getitem_212, getitem_213]);  getitem_211 = getitem_212 = getitem_213 = None
    view_175: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.reshape.default(cat_6, [3, 32, 4, 64, 48]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_156: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_175, [1, 3, 0, 2, 4]);  view_175 = None
    clone_74: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_176: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(clone_74, [32, 64, 576]);  clone_74 = None
    view_177: "f32[2048, 576]" = torch.ops.aten.reshape.default(view_176, [2048, 576]);  view_176 = None
    mm_32: "f32[2048, 192]" = torch.ops.aten.mm.default(view_177, permute_157);  permute_157 = None
    permute_158: "f32[576, 2048]" = torch.ops.aten.permute.default(view_177, [1, 0])
    mm_33: "f32[576, 192]" = torch.ops.aten.mm.default(permute_158, view_59);  permute_158 = view_59 = None
    permute_159: "f32[192, 576]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_71: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
    view_178: "f32[576]" = torch.ops.aten.reshape.default(sum_71, [576]);  sum_71 = None
    permute_160: "f32[576, 192]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_179: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_32, [32, 64, 192]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_481: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_179, primals_152);  primals_152 = None
    mul_482: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_481, 192)
    sum_72: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True)
    mul_483: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_481, mul_205);  mul_481 = None
    sum_73: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True);  mul_483 = None
    mul_484: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_205, sum_73);  sum_73 = None
    sub_129: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_482, sum_72);  mul_482 = sum_72 = None
    sub_130: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_129, mul_484);  sub_129 = mul_484 = None
    mul_485: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_10, sub_130);  div_10 = sub_130 = None
    mul_486: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_179, mul_205);  mul_205 = None
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_486, [0, 1]);  mul_486 = None
    sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_242: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_241, mul_485);  add_241 = mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_180: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_242, [2048, 192])
    mm_34: "f32[2048, 384]" = torch.ops.aten.mm.default(view_180, permute_161);  permute_161 = None
    permute_162: "f32[192, 2048]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_35: "f32[192, 384]" = torch.ops.aten.mm.default(permute_162, view_57);  permute_162 = view_57 = None
    permute_163: "f32[384, 192]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_76: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
    view_181: "f32[192]" = torch.ops.aten.reshape.default(sum_76, [192]);  sum_76 = None
    permute_164: "f32[192, 384]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_182: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(mm_34, [32, 64, 384]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_131: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_default_11, sigmoid_21)
    mul_487: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_56, sub_131);  view_56 = sub_131 = None
    add_243: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_487, 1);  mul_487 = None
    mul_488: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_21, add_243);  sigmoid_21 = add_243 = None
    mul_489: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_182, mul_488);  view_182 = mul_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_183: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_489, [2048, 384]);  mul_489 = None
    mm_36: "f32[2048, 192]" = torch.ops.aten.mm.default(view_183, permute_166);  permute_166 = None
    permute_167: "f32[384, 2048]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_37: "f32[384, 192]" = torch.ops.aten.mm.default(permute_167, view_55);  permute_167 = view_55 = None
    permute_168: "f32[192, 384]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_77: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[384]" = torch.ops.aten.reshape.default(sum_77, [384]);  sum_77 = None
    permute_169: "f32[384, 192]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_185: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_36, [32, 64, 192]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_491: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_185, primals_146);  primals_146 = None
    mul_492: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_491, 192)
    sum_78: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True)
    mul_493: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_491, mul_202);  mul_491 = None
    sum_79: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
    mul_494: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_202, sum_79);  sum_79 = None
    sub_133: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_492, sum_78);  mul_492 = sum_78 = None
    sub_134: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_133, mul_494);  sub_133 = mul_494 = None
    mul_495: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_11, sub_134);  div_11 = sub_134 = None
    mul_496: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_185, mul_202);  mul_202 = None
    sum_80: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1]);  mul_496 = None
    sum_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_244: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_242, mul_495);  add_242 = mul_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_186: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_244, [2048, 192])
    mm_38: "f32[2048, 192]" = torch.ops.aten.mm.default(view_186, permute_170);  permute_170 = None
    permute_171: "f32[192, 2048]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_39: "f32[192, 192]" = torch.ops.aten.mm.default(permute_171, view_53);  permute_171 = view_53 = None
    permute_172: "f32[192, 192]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_82: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[192]" = torch.ops.aten.reshape.default(sum_82, [192]);  sum_82 = None
    permute_173: "f32[192, 192]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_188: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_38, [32, 64, 192]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_189: "f32[32, 64, 4, 48]" = torch.ops.aten.reshape.default(view_188, [32, 64, 4, 48]);  view_188 = None
    permute_174: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_174, getitem_94, getitem_95, getitem_96, None, alias_13, getitem_98, getitem_99, getitem_100, 0.0, [True, True, True, False]);  permute_174 = getitem_94 = getitem_95 = getitem_96 = alias_13 = getitem_98 = getitem_99 = getitem_100 = None
    getitem_215: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_216: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_217: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_215, getitem_216, getitem_217]);  getitem_215 = getitem_216 = getitem_217 = None
    view_190: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.reshape.default(cat_7, [3, 32, 4, 64, 48]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_175: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_190, [1, 3, 0, 2, 4]);  view_190 = None
    clone_75: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_191: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(clone_75, [32, 64, 576]);  clone_75 = None
    view_192: "f32[2048, 576]" = torch.ops.aten.reshape.default(view_191, [2048, 576]);  view_191 = None
    mm_40: "f32[2048, 192]" = torch.ops.aten.mm.default(view_192, permute_176);  permute_176 = None
    permute_177: "f32[576, 2048]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_41: "f32[576, 192]" = torch.ops.aten.mm.default(permute_177, view_49);  permute_177 = view_49 = None
    permute_178: "f32[192, 576]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_83: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
    view_193: "f32[576]" = torch.ops.aten.reshape.default(sum_83, [576]);  sum_83 = None
    permute_179: "f32[576, 192]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_194: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_40, [32, 64, 192]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_498: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_194, primals_140);  primals_140 = None
    mul_499: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_498, 192)
    sum_84: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True)
    mul_500: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_498, mul_200);  mul_498 = None
    sum_85: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [2], True);  mul_500 = None
    mul_501: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_200, sum_85);  sum_85 = None
    sub_136: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_499, sum_84);  mul_499 = sum_84 = None
    sub_137: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_136, mul_501);  sub_136 = mul_501 = None
    mul_502: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_12, sub_137);  div_12 = sub_137 = None
    mul_503: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_194, mul_200);  mul_200 = None
    sum_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 1]);  mul_503 = None
    sum_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_194, [0, 1]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_245: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_244, mul_502);  add_244 = mul_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_195: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_245, [2048, 192])
    mm_42: "f32[2048, 384]" = torch.ops.aten.mm.default(view_195, permute_180);  permute_180 = None
    permute_181: "f32[192, 2048]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_43: "f32[192, 384]" = torch.ops.aten.mm.default(permute_181, view_47);  permute_181 = view_47 = None
    permute_182: "f32[384, 192]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_88: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
    view_196: "f32[192]" = torch.ops.aten.reshape.default(sum_88, [192]);  sum_88 = None
    permute_183: "f32[192, 384]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_197: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(mm_42, [32, 64, 384]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_138: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_default_11, sigmoid_20)
    mul_504: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_46, sub_138);  view_46 = sub_138 = None
    add_246: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_504, 1);  mul_504 = None
    mul_505: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_20, add_246);  sigmoid_20 = add_246 = None
    mul_506: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_197, mul_505);  view_197 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_198: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_506, [2048, 384]);  mul_506 = None
    mm_44: "f32[2048, 192]" = torch.ops.aten.mm.default(view_198, permute_185);  permute_185 = None
    permute_186: "f32[384, 2048]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_45: "f32[384, 192]" = torch.ops.aten.mm.default(permute_186, view_45);  permute_186 = view_45 = None
    permute_187: "f32[192, 384]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_89: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[384]" = torch.ops.aten.reshape.default(sum_89, [384]);  sum_89 = None
    permute_188: "f32[384, 192]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    view_200: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_44, [32, 64, 192]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_508: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_200, primals_134);  primals_134 = None
    mul_509: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_508, 192)
    sum_90: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True)
    mul_510: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_508, mul_197);  mul_508 = None
    sum_91: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True);  mul_510 = None
    mul_511: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_197, sum_91);  sum_91 = None
    sub_140: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_509, sum_90);  mul_509 = sum_90 = None
    sub_141: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_140, mul_511);  sub_140 = mul_511 = None
    mul_512: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_13, sub_141);  div_13 = sub_141 = None
    mul_513: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_200, mul_197);  mul_197 = None
    sum_92: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 1]);  mul_513 = None
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_200, [0, 1]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_247: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_245, mul_512);  add_245 = mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_201: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_247, [2048, 192])
    mm_46: "f32[2048, 192]" = torch.ops.aten.mm.default(view_201, permute_189);  permute_189 = None
    permute_190: "f32[192, 2048]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_47: "f32[192, 192]" = torch.ops.aten.mm.default(permute_190, view_43);  permute_190 = view_43 = None
    permute_191: "f32[192, 192]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_94: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[192]" = torch.ops.aten.reshape.default(sum_94, [192]);  sum_94 = None
    permute_192: "f32[192, 192]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    view_203: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_46, [32, 64, 192]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_204: "f32[32, 64, 4, 48]" = torch.ops.aten.reshape.default(view_203, [32, 64, 4, 48]);  view_203 = None
    permute_193: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_193, getitem_83, getitem_84, getitem_85, None, alias_14, getitem_87, getitem_88, getitem_89, 0.0, [True, True, True, False]);  permute_193 = getitem_83 = getitem_84 = getitem_85 = alias_14 = getitem_87 = getitem_88 = getitem_89 = None
    getitem_219: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_220: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_221: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_219, getitem_220, getitem_221]);  getitem_219 = getitem_220 = getitem_221 = None
    view_205: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.reshape.default(cat_8, [3, 32, 4, 64, 48]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_194: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_205, [1, 3, 0, 2, 4]);  view_205 = None
    clone_76: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_206: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(clone_76, [32, 64, 576]);  clone_76 = None
    view_207: "f32[2048, 576]" = torch.ops.aten.reshape.default(view_206, [2048, 576]);  view_206 = None
    mm_48: "f32[2048, 192]" = torch.ops.aten.mm.default(view_207, permute_195);  permute_195 = None
    permute_196: "f32[576, 2048]" = torch.ops.aten.permute.default(view_207, [1, 0])
    mm_49: "f32[576, 192]" = torch.ops.aten.mm.default(permute_196, view_39);  permute_196 = view_39 = None
    permute_197: "f32[192, 576]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_95: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
    view_208: "f32[576]" = torch.ops.aten.reshape.default(sum_95, [576]);  sum_95 = None
    permute_198: "f32[576, 192]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_209: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_48, [32, 64, 192]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_515: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_209, primals_128);  primals_128 = None
    mul_516: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_515, 192)
    sum_96: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True)
    mul_517: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_515, mul_195);  mul_515 = None
    sum_97: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True);  mul_517 = None
    mul_518: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_195, sum_97);  sum_97 = None
    sub_143: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_516, sum_96);  mul_516 = sum_96 = None
    sub_144: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_143, mul_518);  sub_143 = mul_518 = None
    mul_519: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_14, sub_144);  div_14 = sub_144 = None
    mul_520: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_209, mul_195);  mul_195 = None
    sum_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 1]);  mul_520 = None
    sum_99: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_209, [0, 1]);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_248: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_247, mul_519);  add_247 = mul_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_210: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_248, [2048, 192])
    mm_50: "f32[2048, 384]" = torch.ops.aten.mm.default(view_210, permute_199);  permute_199 = None
    permute_200: "f32[192, 2048]" = torch.ops.aten.permute.default(view_210, [1, 0])
    mm_51: "f32[192, 384]" = torch.ops.aten.mm.default(permute_200, view_37);  permute_200 = view_37 = None
    permute_201: "f32[384, 192]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_100: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[192]" = torch.ops.aten.reshape.default(sum_100, [192]);  sum_100 = None
    permute_202: "f32[192, 384]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_212: "f32[32, 64, 384]" = torch.ops.aten.reshape.default(mm_50, [32, 64, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_145: "f32[32, 64, 384]" = torch.ops.aten.sub.Tensor(full_default_11, sigmoid_19);  full_default_11 = None
    mul_521: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_36, sub_145);  view_36 = sub_145 = None
    add_249: "f32[32, 64, 384]" = torch.ops.aten.add.Scalar(mul_521, 1);  mul_521 = None
    mul_522: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(sigmoid_19, add_249);  sigmoid_19 = add_249 = None
    mul_523: "f32[32, 64, 384]" = torch.ops.aten.mul.Tensor(view_212, mul_522);  view_212 = mul_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[2048, 384]" = torch.ops.aten.reshape.default(mul_523, [2048, 384]);  mul_523 = None
    mm_52: "f32[2048, 192]" = torch.ops.aten.mm.default(view_213, permute_204);  permute_204 = None
    permute_205: "f32[384, 2048]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_53: "f32[384, 192]" = torch.ops.aten.mm.default(permute_205, view_35);  permute_205 = view_35 = None
    permute_206: "f32[192, 384]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_101: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[384]" = torch.ops.aten.reshape.default(sum_101, [384]);  sum_101 = None
    permute_207: "f32[384, 192]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_215: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_52, [32, 64, 192]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_525: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_215, primals_122);  primals_122 = None
    mul_526: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_525, 192)
    sum_102: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True)
    mul_527: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_525, mul_192);  mul_525 = None
    sum_103: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True);  mul_527 = None
    mul_528: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_192, sum_103);  sum_103 = None
    sub_147: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_526, sum_102);  mul_526 = sum_102 = None
    sub_148: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_147, mul_528);  sub_147 = mul_528 = None
    mul_529: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_15, sub_148);  div_15 = sub_148 = None
    mul_530: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_215, mul_192);  mul_192 = None
    sum_104: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 1]);  mul_530 = None
    sum_105: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_215, [0, 1]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_250: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_248, mul_529);  add_248 = mul_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_216: "f32[2048, 192]" = torch.ops.aten.reshape.default(add_250, [2048, 192])
    mm_54: "f32[2048, 192]" = torch.ops.aten.mm.default(view_216, permute_208);  permute_208 = None
    permute_209: "f32[192, 2048]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_55: "f32[192, 192]" = torch.ops.aten.mm.default(permute_209, view_33);  permute_209 = view_33 = None
    permute_210: "f32[192, 192]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_106: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
    view_217: "f32[192]" = torch.ops.aten.reshape.default(sum_106, [192]);  sum_106 = None
    permute_211: "f32[192, 192]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_218: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_54, [32, 64, 192]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_219: "f32[32, 64, 4, 48]" = torch.ops.aten.reshape.default(view_218, [32, 64, 4, 48]);  view_218 = None
    permute_212: "f32[32, 4, 64, 48]" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_212, getitem_72, getitem_73, getitem_74, None, alias_15, getitem_76, getitem_77, getitem_78, 0.0, [True, True, True, False]);  permute_212 = getitem_72 = getitem_73 = getitem_74 = alias_15 = getitem_76 = getitem_77 = getitem_78 = None
    getitem_223: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_224: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_225: "f32[32, 4, 64, 48]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[96, 4, 64, 48]" = torch.ops.aten.cat.default([getitem_223, getitem_224, getitem_225]);  getitem_223 = getitem_224 = getitem_225 = None
    view_220: "f32[3, 32, 4, 64, 48]" = torch.ops.aten.reshape.default(cat_9, [3, 32, 4, 64, 48]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_213: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.permute.default(view_220, [1, 3, 0, 2, 4]);  view_220 = None
    clone_77: "f32[32, 64, 3, 4, 48]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    view_221: "f32[32, 64, 576]" = torch.ops.aten.reshape.default(clone_77, [32, 64, 576]);  clone_77 = None
    view_222: "f32[2048, 576]" = torch.ops.aten.reshape.default(view_221, [2048, 576]);  view_221 = None
    mm_56: "f32[2048, 192]" = torch.ops.aten.mm.default(view_222, permute_214);  permute_214 = None
    permute_215: "f32[576, 2048]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_57: "f32[576, 192]" = torch.ops.aten.mm.default(permute_215, view_29);  permute_215 = view_29 = None
    permute_216: "f32[192, 576]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_107: "f32[1, 576]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[576]" = torch.ops.aten.reshape.default(sum_107, [576]);  sum_107 = None
    permute_217: "f32[576, 192]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    view_224: "f32[32, 64, 192]" = torch.ops.aten.reshape.default(mm_56, [32, 64, 192]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_532: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_224, primals_116);  primals_116 = None
    mul_533: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_532, 192)
    sum_108: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_532, [2], True)
    mul_534: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_532, mul_190);  mul_532 = None
    sum_109: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True);  mul_534 = None
    mul_535: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(mul_190, sum_109);  sum_109 = None
    sub_150: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(mul_533, sum_108);  mul_533 = sum_108 = None
    sub_151: "f32[32, 64, 192]" = torch.ops.aten.sub.Tensor(sub_150, mul_535);  sub_150 = mul_535 = None
    mul_536: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(div_16, sub_151);  div_16 = sub_151 = None
    mul_537: "f32[32, 64, 192]" = torch.ops.aten.mul.Tensor(view_224, mul_190);  mul_190 = None
    sum_110: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 1]);  mul_537 = None
    sum_111: "f32[192]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_251: "f32[32, 64, 192]" = torch.ops.aten.add.Tensor(add_250, mul_536);  add_250 = mul_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_225: "f32[8, 4, 64, 192]" = torch.ops.aten.reshape.default(add_251, [8, 4, 64, 192]);  add_251 = None
    permute_218: "f32[8, 192, 64, 4]" = torch.ops.aten.permute.default(view_225, [0, 3, 2, 1]);  view_225 = None
    clone_78: "f32[8, 192, 64, 4]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_226: "f32[12288, 8, 2, 2]" = torch.ops.aten.reshape.default(clone_78, [12288, 8, 2, 2]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_219: "f32[12288, 2, 8, 2]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_79: "f32[12288, 2, 8, 2]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_227: "f32[8, 192, 16, 16]" = torch.ops.aten.reshape.default(clone_79, [8, 192, 16, 16]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_227, mul_189, primals_115, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_227 = mul_189 = primals_115 = None
    getitem_227: "f32[8, 128, 16, 16]" = convolution_backward_10[0]
    getitem_228: "f32[192, 128, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_540: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_227, mul_539);  getitem_227 = mul_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 2, 3])
    sub_153: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_238);  convolution_23 = unsqueeze_238 = None
    mul_541: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(mul_540, sub_153)
    sum_113: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 2, 3]);  mul_541 = None
    mul_542: "f32[128]" = torch.ops.aten.mul.Tensor(sum_112, 0.00048828125)
    unsqueeze_239: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_240: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_543: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, 0.00048828125)
    mul_544: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_545: "f32[128]" = torch.ops.aten.mul.Tensor(mul_543, mul_544);  mul_543 = mul_544 = None
    unsqueeze_242: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_243: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_546: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_245: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_246: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    mul_547: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_244);  sub_153 = unsqueeze_244 = None
    sub_155: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(mul_540, mul_547);  mul_540 = mul_547 = None
    sub_156: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_241);  sub_155 = unsqueeze_241 = None
    mul_548: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_247);  sub_156 = unsqueeze_247 = None
    mul_549: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_67);  sum_113 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_548, add_125, primals_114, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_548 = add_125 = primals_114 = None
    getitem_230: "f32[8, 128, 16, 16]" = convolution_backward_11[0]
    getitem_231: "f32[128, 128, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_253: "f32[8, 128, 16, 16]" = torch.ops.aten.add.Tensor(slice_3, getitem_230);  slice_3 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_253, [0, 2, 3])
    sub_157: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_250);  convolution_22 = unsqueeze_250 = None
    mul_550: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(add_253, sub_157)
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 2, 3]);  mul_550 = None
    mul_551: "f32[128]" = torch.ops.aten.mul.Tensor(sum_114, 0.00048828125)
    unsqueeze_251: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_252: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_552: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, 0.00048828125)
    mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(mul_552, mul_553);  mul_552 = mul_553 = None
    unsqueeze_254: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_255: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_257: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_258: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_556: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_256);  sub_157 = unsqueeze_256 = None
    sub_159: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(add_253, mul_556);  add_253 = mul_556 = None
    sub_160: "f32[8, 128, 16, 16]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_253);  sub_159 = unsqueeze_253 = None
    mul_557: "f32[8, 128, 16, 16]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_259);  sub_160 = unsqueeze_259 = None
    mul_558: "f32[128]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_64);  sum_115 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_557, mul_174, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = mul_174 = primals_113 = None
    getitem_233: "f32[8, 384, 16, 16]" = convolution_backward_12[0]
    getitem_234: "f32[128, 384, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_561: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(getitem_233, mul_560);  getitem_233 = mul_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3])
    sub_162: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_262);  convolution_21 = unsqueeze_262 = None
    mul_562: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(mul_561, sub_162)
    sum_117: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_563: "f32[384]" = torch.ops.aten.mul.Tensor(sum_116, 0.00048828125)
    unsqueeze_263: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_264: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(sum_117, 0.00048828125)
    mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_266: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_267: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_269: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_270: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_568: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_268);  sub_162 = unsqueeze_268 = None
    sub_164: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(mul_561, mul_568);  mul_561 = mul_568 = None
    sub_165: "f32[8, 384, 16, 16]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_265);  sub_164 = unsqueeze_265 = None
    mul_569: "f32[8, 384, 16, 16]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_271);  sub_165 = unsqueeze_271 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_61);  sum_117 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_569, mul_166, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_569 = mul_166 = primals_112 = None
    getitem_236: "f32[8, 384, 32, 32]" = convolution_backward_13[0]
    getitem_237: "f32[384, 1, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_573: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_236, mul_572);  getitem_236 = mul_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3])
    sub_167: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_274);  convolution_20 = unsqueeze_274 = None
    mul_574: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(mul_573, sub_167)
    sum_119: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(sum_118, 0.0001220703125)
    unsqueeze_275: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_276: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(sum_119, 0.0001220703125)
    mul_577: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_578: "f32[384]" = torch.ops.aten.mul.Tensor(mul_576, mul_577);  mul_576 = mul_577 = None
    unsqueeze_278: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
    unsqueeze_279: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_281: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_282: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_580: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_280);  sub_167 = unsqueeze_280 = None
    sub_169: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(mul_573, mul_580);  mul_573 = mul_580 = None
    sub_170: "f32[8, 384, 32, 32]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_277);  sub_169 = unsqueeze_277 = None
    mul_581: "f32[8, 384, 32, 32]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_283);  sub_170 = unsqueeze_283 = None
    mul_582: "f32[384]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_58);  sum_119 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_581, mul_158, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_581 = mul_158 = primals_111 = None
    getitem_239: "f32[8, 96, 32, 32]" = convolution_backward_14[0]
    getitem_240: "f32[384, 96, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_585: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_239, mul_584);  getitem_239 = mul_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 2, 3])
    sub_172: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_286);  convolution_19 = unsqueeze_286 = None
    mul_586: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_585, sub_172)
    sum_121: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_586, [0, 2, 3]);  mul_586 = None
    mul_587: "f32[96]" = torch.ops.aten.mul.Tensor(sum_120, 0.0001220703125)
    unsqueeze_287: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
    unsqueeze_288: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_588: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, 0.0001220703125)
    mul_589: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_590: "f32[96]" = torch.ops.aten.mul.Tensor(mul_588, mul_589);  mul_588 = mul_589 = None
    unsqueeze_290: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_291: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_591: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_293: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_294: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_592: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_292);  sub_172 = unsqueeze_292 = None
    sub_174: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_585, mul_592);  mul_585 = mul_592 = None
    sub_175: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_289);  sub_174 = unsqueeze_289 = None
    mul_593: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_295);  sub_175 = unsqueeze_295 = None
    mul_594: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_55);  sum_121 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_593, cat, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_593 = cat = primals_110 = None
    getitem_242: "f32[8, 192, 32, 32]" = convolution_backward_15[0]
    getitem_243: "f32[96, 192, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:268, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
    slice_5: "f32[8, 96, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_242, 1, 0, 96)
    slice_6: "f32[8, 96, 32, 32]" = torch.ops.aten.slice.Tensor(getitem_242, 1, 96, 192);  getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_597: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(slice_6, mul_596);  slice_6 = mul_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3])
    sub_177: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_298);  convolution_18 = unsqueeze_298 = None
    mul_598: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_597, sub_177)
    sum_123: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
    mul_599: "f32[96]" = torch.ops.aten.mul.Tensor(sum_122, 0.0001220703125)
    unsqueeze_299: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_300: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_600: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, 0.0001220703125)
    mul_601: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_602: "f32[96]" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
    unsqueeze_302: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_303: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_603: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_305: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_306: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_604: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_304);  sub_177 = unsqueeze_304 = None
    sub_179: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_597, mul_604);  mul_597 = mul_604 = None
    sub_180: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_301);  sub_179 = unsqueeze_301 = None
    mul_605: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_307);  sub_180 = unsqueeze_307 = None
    mul_606: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_52);  sum_123 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_605, view_25, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_605 = view_25 = primals_109 = None
    getitem_245: "f32[8, 144, 32, 32]" = convolution_backward_16[0]
    getitem_246: "f32[96, 144, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:262, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
    view_228: "f32[18432, 2, 16, 2]" = torch.ops.aten.reshape.default(getitem_245, [18432, 2, 16, 2]);  getitem_245 = None
    permute_225: "f32[18432, 16, 2, 2]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:260, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
    clone_80: "f32[18432, 16, 2, 2]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_229: "f32[8, 144, 256, 4]" = torch.ops.aten.reshape.default(clone_80, [8, 144, 256, 4]);  clone_80 = None
    permute_226: "f32[8, 4, 256, 144]" = torch.ops.aten.permute.default(view_229, [0, 3, 2, 1]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:259, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
    clone_81: "f32[8, 4, 256, 144]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_230: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(clone_81, [32, 256, 144]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:255, code: x = self.norm(x)
    mul_608: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_230, primals_107);  primals_107 = None
    mul_609: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_608, 144)
    sum_124: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_608, [2], True)
    mul_610: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_608, mul_141);  mul_608 = None
    sum_125: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_610, [2], True);  mul_610 = None
    mul_611: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_141, sum_125);  sum_125 = None
    sub_182: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_609, sum_124);  mul_609 = sum_124 = None
    sub_183: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_182, mul_611);  sub_182 = mul_611 = None
    mul_612: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_17, sub_183);  div_17 = sub_183 = None
    mul_613: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_230, mul_141);  mul_141 = None
    sum_126: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_613, [0, 1]);  mul_613 = None
    sum_127: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_230, [0, 1]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_231: "f32[8192, 144]" = torch.ops.aten.reshape.default(mul_612, [8192, 144])
    mm_58: "f32[8192, 288]" = torch.ops.aten.mm.default(view_231, permute_227);  permute_227 = None
    permute_228: "f32[144, 8192]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_59: "f32[144, 288]" = torch.ops.aten.mm.default(permute_228, view_21);  permute_228 = view_21 = None
    permute_229: "f32[288, 144]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_128: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[144]" = torch.ops.aten.reshape.default(sum_128, [144]);  sum_128 = None
    permute_230: "f32[144, 288]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_233: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(mm_58, [32, 256, 288]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    full_default_20: "f32[32, 256, 288]" = torch.ops.aten.full.default([32, 256, 288], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sub_184: "f32[32, 256, 288]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_13)
    mul_614: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_20, sub_184);  view_20 = sub_184 = None
    add_258: "f32[32, 256, 288]" = torch.ops.aten.add.Scalar(mul_614, 1);  mul_614 = None
    mul_615: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(sigmoid_13, add_258);  sigmoid_13 = add_258 = None
    mul_616: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_233, mul_615);  view_233 = mul_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_234: "f32[8192, 288]" = torch.ops.aten.reshape.default(mul_616, [8192, 288]);  mul_616 = None
    mm_60: "f32[8192, 144]" = torch.ops.aten.mm.default(view_234, permute_232);  permute_232 = None
    permute_233: "f32[288, 8192]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_61: "f32[288, 144]" = torch.ops.aten.mm.default(permute_233, view_19);  permute_233 = view_19 = None
    permute_234: "f32[144, 288]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_129: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[288]" = torch.ops.aten.reshape.default(sum_129, [288]);  sum_129 = None
    permute_235: "f32[288, 144]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_236: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_60, [32, 256, 144]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_618: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_236, primals_101);  primals_101 = None
    mul_619: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_618, 144)
    sum_130: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True)
    mul_620: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_618, mul_138);  mul_618 = None
    sum_131: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
    mul_621: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_138, sum_131);  sum_131 = None
    sub_186: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_619, sum_130);  mul_619 = sum_130 = None
    sub_187: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_186, mul_621);  sub_186 = mul_621 = None
    mul_622: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_18, sub_187);  div_18 = sub_187 = None
    mul_623: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_236, mul_138);  mul_138 = None
    sum_132: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1]);  mul_623 = None
    sum_133: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_236, [0, 1]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_259: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(mul_612, mul_622);  mul_612 = mul_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_237: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_259, [8192, 144])
    mm_62: "f32[8192, 144]" = torch.ops.aten.mm.default(view_237, permute_236);  permute_236 = None
    permute_237: "f32[144, 8192]" = torch.ops.aten.permute.default(view_237, [1, 0])
    mm_63: "f32[144, 144]" = torch.ops.aten.mm.default(permute_237, view_17);  permute_237 = view_17 = None
    permute_238: "f32[144, 144]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_134: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
    view_238: "f32[144]" = torch.ops.aten.reshape.default(sum_134, [144]);  sum_134 = None
    permute_239: "f32[144, 144]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_239: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_62, [32, 256, 144]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_240: "f32[32, 256, 4, 36]" = torch.ops.aten.reshape.default(view_239, [32, 256, 4, 36]);  view_239 = None
    permute_240: "f32[32, 4, 256, 36]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_240, getitem_47, getitem_48, getitem_49, None, alias_16, getitem_51, getitem_52, getitem_53, 0.0, [True, True, True, False]);  permute_240 = getitem_47 = getitem_48 = getitem_49 = alias_16 = getitem_51 = getitem_52 = getitem_53 = None
    getitem_248: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_249: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_250: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[96, 4, 256, 36]" = torch.ops.aten.cat.default([getitem_248, getitem_249, getitem_250]);  getitem_248 = getitem_249 = getitem_250 = None
    view_241: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.reshape.default(cat_10, [3, 32, 4, 256, 36]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_241: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.permute.default(view_241, [1, 3, 0, 2, 4]);  view_241 = None
    clone_82: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_242: "f32[32, 256, 432]" = torch.ops.aten.reshape.default(clone_82, [32, 256, 432]);  clone_82 = None
    view_243: "f32[8192, 432]" = torch.ops.aten.reshape.default(view_242, [8192, 432]);  view_242 = None
    mm_64: "f32[8192, 144]" = torch.ops.aten.mm.default(view_243, permute_242);  permute_242 = None
    permute_243: "f32[432, 8192]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_65: "f32[432, 144]" = torch.ops.aten.mm.default(permute_243, view_13);  permute_243 = view_13 = None
    permute_244: "f32[144, 432]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_135: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[432]" = torch.ops.aten.reshape.default(sum_135, [432]);  sum_135 = None
    permute_245: "f32[432, 144]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_245: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_64, [32, 256, 144]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_625: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_245, primals_95);  primals_95 = None
    mul_626: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_625, 144)
    sum_136: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True)
    mul_627: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_625, mul_136);  mul_625 = None
    sum_137: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True);  mul_627 = None
    mul_628: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_136, sum_137);  sum_137 = None
    sub_189: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_626, sum_136);  mul_626 = sum_136 = None
    sub_190: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_189, mul_628);  sub_189 = mul_628 = None
    mul_629: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_19, sub_190);  div_19 = sub_190 = None
    mul_630: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_245, mul_136);  mul_136 = None
    sum_138: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 1]);  mul_630 = None
    sum_139: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_260: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_259, mul_629);  add_259 = mul_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_260, [8192, 144])
    mm_66: "f32[8192, 288]" = torch.ops.aten.mm.default(view_246, permute_246);  permute_246 = None
    permute_247: "f32[144, 8192]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_67: "f32[144, 288]" = torch.ops.aten.mm.default(permute_247, view_11);  permute_247 = view_11 = None
    permute_248: "f32[288, 144]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_140: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[144]" = torch.ops.aten.reshape.default(sum_140, [144]);  sum_140 = None
    permute_249: "f32[144, 288]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_248: "f32[32, 256, 288]" = torch.ops.aten.reshape.default(mm_66, [32, 256, 288]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    sub_191: "f32[32, 256, 288]" = torch.ops.aten.sub.Tensor(full_default_20, sigmoid_12);  full_default_20 = None
    mul_631: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_10, sub_191);  view_10 = sub_191 = None
    add_261: "f32[32, 256, 288]" = torch.ops.aten.add.Scalar(mul_631, 1);  mul_631 = None
    mul_632: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(sigmoid_12, add_261);  sigmoid_12 = add_261 = None
    mul_633: "f32[32, 256, 288]" = torch.ops.aten.mul.Tensor(view_248, mul_632);  view_248 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[8192, 288]" = torch.ops.aten.reshape.default(mul_633, [8192, 288]);  mul_633 = None
    mm_68: "f32[8192, 144]" = torch.ops.aten.mm.default(view_249, permute_251);  permute_251 = None
    permute_252: "f32[288, 8192]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_69: "f32[288, 144]" = torch.ops.aten.mm.default(permute_252, view_9);  permute_252 = view_9 = None
    permute_253: "f32[144, 288]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_141: "f32[1, 288]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[288]" = torch.ops.aten.reshape.default(sum_141, [288]);  sum_141 = None
    permute_254: "f32[288, 144]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    view_251: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_68, [32, 256, 144]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_635: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_251, primals_89);  primals_89 = None
    mul_636: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_635, 144)
    sum_142: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
    mul_637: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_635, mul_133);  mul_635 = None
    sum_143: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
    mul_638: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_133, sum_143);  sum_143 = None
    sub_193: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_636, sum_142);  mul_636 = sum_142 = None
    sub_194: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_193, mul_638);  sub_193 = mul_638 = None
    mul_639: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_20, sub_194);  div_20 = sub_194 = None
    mul_640: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_251, mul_133);  mul_133 = None
    sum_144: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
    sum_145: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_262: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_260, mul_639);  add_260 = mul_639 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_252: "f32[8192, 144]" = torch.ops.aten.reshape.default(add_262, [8192, 144])
    mm_70: "f32[8192, 144]" = torch.ops.aten.mm.default(view_252, permute_255);  permute_255 = None
    permute_256: "f32[144, 8192]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_71: "f32[144, 144]" = torch.ops.aten.mm.default(permute_256, view_7);  permute_256 = view_7 = None
    permute_257: "f32[144, 144]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_146: "f32[1, 144]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[144]" = torch.ops.aten.reshape.default(sum_146, [144]);  sum_146 = None
    permute_258: "f32[144, 144]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_254: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_70, [32, 256, 144]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_255: "f32[32, 256, 4, 36]" = torch.ops.aten.reshape.default(view_254, [32, 256, 4, 36]);  view_254 = None
    permute_259: "f32[32, 4, 256, 36]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_259, getitem_36, getitem_37, getitem_38, None, alias_17, getitem_40, getitem_41, getitem_42, 0.0, [True, True, True, False]);  permute_259 = getitem_36 = getitem_37 = getitem_38 = alias_17 = getitem_40 = getitem_41 = getitem_42 = None
    getitem_252: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_253: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_254: "f32[32, 4, 256, 36]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[96, 4, 256, 36]" = torch.ops.aten.cat.default([getitem_252, getitem_253, getitem_254]);  getitem_252 = getitem_253 = getitem_254 = None
    view_256: "f32[3, 32, 4, 256, 36]" = torch.ops.aten.reshape.default(cat_11, [3, 32, 4, 256, 36]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_260: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 2, 4]);  view_256 = None
    clone_83: "f32[32, 256, 3, 4, 36]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_257: "f32[32, 256, 432]" = torch.ops.aten.reshape.default(clone_83, [32, 256, 432]);  clone_83 = None
    view_258: "f32[8192, 432]" = torch.ops.aten.reshape.default(view_257, [8192, 432]);  view_257 = None
    mm_72: "f32[8192, 144]" = torch.ops.aten.mm.default(view_258, permute_261);  permute_261 = None
    permute_262: "f32[432, 8192]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_73: "f32[432, 144]" = torch.ops.aten.mm.default(permute_262, view_3);  permute_262 = view_3 = None
    permute_263: "f32[144, 432]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_147: "f32[1, 432]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[432]" = torch.ops.aten.reshape.default(sum_147, [432]);  sum_147 = None
    permute_264: "f32[432, 144]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_260: "f32[32, 256, 144]" = torch.ops.aten.reshape.default(mm_72, [32, 256, 144]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_642: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_260, primals_83);  primals_83 = None
    mul_643: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_642, 144)
    sum_148: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [2], True)
    mul_644: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_642, mul_131);  mul_642 = None
    sum_149: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True);  mul_644 = None
    mul_645: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(mul_131, sum_149);  sum_149 = None
    sub_196: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(mul_643, sum_148);  mul_643 = sum_148 = None
    sub_197: "f32[32, 256, 144]" = torch.ops.aten.sub.Tensor(sub_196, mul_645);  sub_196 = mul_645 = None
    mul_646: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(div_21, sub_197);  div_21 = sub_197 = None
    mul_647: "f32[32, 256, 144]" = torch.ops.aten.mul.Tensor(view_260, mul_131);  mul_131 = None
    sum_150: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 1]);  mul_647 = None
    sum_151: "f32[144]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_263: "f32[32, 256, 144]" = torch.ops.aten.add.Tensor(add_262, mul_646);  add_262 = mul_646 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:251, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
    view_261: "f32[8, 4, 256, 144]" = torch.ops.aten.reshape.default(add_263, [8, 4, 256, 144]);  add_263 = None
    permute_265: "f32[8, 144, 256, 4]" = torch.ops.aten.permute.default(view_261, [0, 3, 2, 1]);  view_261 = None
    clone_84: "f32[8, 144, 256, 4]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_262: "f32[18432, 16, 2, 2]" = torch.ops.aten.reshape.default(clone_84, [18432, 16, 2, 2]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:249, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
    permute_266: "f32[18432, 2, 16, 2]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    clone_85: "f32[18432, 2, 16, 2]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_263: "f32[8, 144, 32, 32]" = torch.ops.aten.reshape.default(clone_85, [8, 144, 32, 32]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mobilevit.py:234, code: x = self.conv_1x1(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_263, mul_130, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_263 = mul_130 = primals_82 = None
    getitem_256: "f32[8, 96, 32, 32]" = convolution_backward_17[0]
    getitem_257: "f32[144, 96, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_650: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_256, mul_649);  getitem_256 = mul_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_152: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3])
    sub_199: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_310);  convolution_16 = unsqueeze_310 = None
    mul_651: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(mul_650, sub_199)
    sum_153: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_652: "f32[96]" = torch.ops.aten.mul.Tensor(sum_152, 0.0001220703125)
    unsqueeze_311: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_312: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_653: "f32[96]" = torch.ops.aten.mul.Tensor(sum_153, 0.0001220703125)
    mul_654: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_655: "f32[96]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
    unsqueeze_314: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_315: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_656: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_317: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_318: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_657: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_316);  sub_199 = unsqueeze_316 = None
    sub_201: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(mul_650, mul_657);  mul_650 = mul_657 = None
    sub_202: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_313);  sub_201 = unsqueeze_313 = None
    mul_658: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_319);  sub_202 = unsqueeze_319 = None
    mul_659: "f32[96]" = torch.ops.aten.mul.Tensor(sum_153, squeeze_49);  sum_153 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_658, add_81, primals_81, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_658 = add_81 = primals_81 = None
    getitem_259: "f32[8, 96, 32, 32]" = convolution_backward_18[0]
    getitem_260: "f32[96, 96, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_265: "f32[8, 96, 32, 32]" = torch.ops.aten.add.Tensor(slice_5, getitem_259);  slice_5 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_154: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_265, [0, 2, 3])
    sub_203: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_322);  convolution_15 = unsqueeze_322 = None
    mul_660: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(add_265, sub_203)
    sum_155: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
    mul_661: "f32[96]" = torch.ops.aten.mul.Tensor(sum_154, 0.0001220703125)
    unsqueeze_323: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_324: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_662: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, 0.0001220703125)
    mul_663: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_664: "f32[96]" = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_326: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_327: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_665: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_329: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_330: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_666: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_328);  sub_203 = unsqueeze_328 = None
    sub_205: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(add_265, mul_666);  add_265 = mul_666 = None
    sub_206: "f32[8, 96, 32, 32]" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_325);  sub_205 = unsqueeze_325 = None
    mul_667: "f32[8, 96, 32, 32]" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_331);  sub_206 = unsqueeze_331 = None
    mul_668: "f32[96]" = torch.ops.aten.mul.Tensor(sum_155, squeeze_46);  sum_155 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_667, mul_115, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = mul_115 = primals_80 = None
    getitem_262: "f32[8, 256, 32, 32]" = convolution_backward_19[0]
    getitem_263: "f32[96, 256, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_671: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(getitem_262, mul_670);  getitem_262 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_156: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3])
    sub_208: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_334);  convolution_14 = unsqueeze_334 = None
    mul_672: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(mul_671, sub_208)
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
    mul_673: "f32[256]" = torch.ops.aten.mul.Tensor(sum_156, 0.0001220703125)
    unsqueeze_335: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_336: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_674: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, 0.0001220703125)
    mul_675: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_676: "f32[256]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
    unsqueeze_338: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_339: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_677: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_341: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_342: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_678: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_340);  sub_208 = unsqueeze_340 = None
    sub_210: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(mul_671, mul_678);  mul_671 = mul_678 = None
    sub_211: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_337);  sub_210 = unsqueeze_337 = None
    mul_679: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_343);  sub_211 = unsqueeze_343 = None
    mul_680: "f32[256]" = torch.ops.aten.mul.Tensor(sum_157, squeeze_43);  sum_157 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_679, mul_107, primals_79, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_679 = mul_107 = primals_79 = None
    getitem_265: "f32[8, 256, 64, 64]" = convolution_backward_20[0]
    getitem_266: "f32[256, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_683: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_265, mul_682);  getitem_265 = mul_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 2, 3])
    sub_213: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_346);  convolution_13 = unsqueeze_346 = None
    mul_684: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_683, sub_213)
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3]);  mul_684 = None
    mul_685: "f32[256]" = torch.ops.aten.mul.Tensor(sum_158, 3.0517578125e-05)
    unsqueeze_347: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_348: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_686: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, 3.0517578125e-05)
    mul_687: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_688: "f32[256]" = torch.ops.aten.mul.Tensor(mul_686, mul_687);  mul_686 = mul_687 = None
    unsqueeze_350: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_351: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_689: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_353: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_354: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_690: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_352);  sub_213 = unsqueeze_352 = None
    sub_215: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_683, mul_690);  mul_683 = mul_690 = None
    sub_216: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_349);  sub_215 = unsqueeze_349 = None
    mul_691: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_355);  sub_216 = unsqueeze_355 = None
    mul_692: "f32[256]" = torch.ops.aten.mul.Tensor(sum_159, squeeze_40);  sum_159 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_691, add_66, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = add_66 = primals_78 = None
    getitem_268: "f32[8, 64, 64, 64]" = convolution_backward_21[0]
    getitem_269: "f32[256, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_160: "f32[64]" = torch.ops.aten.sum.dim_IntList(getitem_268, [0, 2, 3])
    sub_217: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_358);  convolution_12 = unsqueeze_358 = None
    mul_693: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_268, sub_217)
    sum_161: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 2, 3]);  mul_693 = None
    mul_694: "f32[64]" = torch.ops.aten.mul.Tensor(sum_160, 3.0517578125e-05)
    unsqueeze_359: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_360: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_695: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, 3.0517578125e-05)
    mul_696: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_697: "f32[64]" = torch.ops.aten.mul.Tensor(mul_695, mul_696);  mul_695 = mul_696 = None
    unsqueeze_362: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_363: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_698: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_365: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_366: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_699: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_364);  sub_217 = unsqueeze_364 = None
    sub_219: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(getitem_268, mul_699);  mul_699 = None
    sub_220: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_361);  sub_219 = unsqueeze_361 = None
    mul_700: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_367);  sub_220 = unsqueeze_367 = None
    mul_701: "f32[64]" = torch.ops.aten.mul.Tensor(sum_161, squeeze_37);  sum_161 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_700, mul_92, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_700 = mul_92 = primals_77 = None
    getitem_271: "f32[8, 256, 64, 64]" = convolution_backward_22[0]
    getitem_272: "f32[64, 256, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_704: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_271, mul_703);  getitem_271 = mul_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_704, [0, 2, 3])
    sub_222: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_370);  convolution_11 = unsqueeze_370 = None
    mul_705: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_704, sub_222)
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
    mul_706: "f32[256]" = torch.ops.aten.mul.Tensor(sum_162, 3.0517578125e-05)
    unsqueeze_371: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
    unsqueeze_372: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_707: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, 3.0517578125e-05)
    mul_708: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_709: "f32[256]" = torch.ops.aten.mul.Tensor(mul_707, mul_708);  mul_707 = mul_708 = None
    unsqueeze_374: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_375: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_710: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_377: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_378: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_711: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_376);  sub_222 = unsqueeze_376 = None
    sub_224: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_704, mul_711);  mul_704 = mul_711 = None
    sub_225: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_373);  sub_224 = unsqueeze_373 = None
    mul_712: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_379);  sub_225 = unsqueeze_379 = None
    mul_713: "f32[256]" = torch.ops.aten.mul.Tensor(sum_163, squeeze_34);  sum_163 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_712, mul_84, primals_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_712 = mul_84 = primals_76 = None
    getitem_274: "f32[8, 256, 64, 64]" = convolution_backward_23[0]
    getitem_275: "f32[256, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_716: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_274, mul_715);  getitem_274 = mul_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3])
    sub_227: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_382);  convolution_10 = unsqueeze_382 = None
    mul_717: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_716, sub_227)
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
    mul_718: "f32[256]" = torch.ops.aten.mul.Tensor(sum_164, 3.0517578125e-05)
    unsqueeze_383: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_384: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_719: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, 3.0517578125e-05)
    mul_720: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_721: "f32[256]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
    unsqueeze_386: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_387: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_722: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_389: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_390: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_723: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_388);  sub_227 = unsqueeze_388 = None
    sub_229: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_716, mul_723);  mul_716 = mul_723 = None
    sub_230: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_385);  sub_229 = unsqueeze_385 = None
    mul_724: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_391);  sub_230 = unsqueeze_391 = None
    mul_725: "f32[256]" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_724, add_50, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = add_50 = primals_75 = None
    getitem_277: "f32[8, 64, 64, 64]" = convolution_backward_24[0]
    getitem_278: "f32[256, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_270: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_268, getitem_277);  getitem_268 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_166: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 2, 3])
    sub_231: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_394);  convolution_9 = unsqueeze_394 = None
    mul_726: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_270, sub_231)
    sum_167: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2, 3]);  mul_726 = None
    mul_727: "f32[64]" = torch.ops.aten.mul.Tensor(sum_166, 3.0517578125e-05)
    unsqueeze_395: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_396: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_728: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, 3.0517578125e-05)
    mul_729: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_730: "f32[64]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
    unsqueeze_398: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_399: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_731: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_401: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_402: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_732: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_400);  sub_231 = unsqueeze_400 = None
    sub_233: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(add_270, mul_732);  mul_732 = None
    sub_234: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_397);  sub_233 = unsqueeze_397 = None
    mul_733: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_403);  sub_234 = unsqueeze_403 = None
    mul_734: "f32[64]" = torch.ops.aten.mul.Tensor(sum_167, squeeze_28);  sum_167 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_733, mul_69, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_733 = mul_69 = primals_74 = None
    getitem_280: "f32[8, 256, 64, 64]" = convolution_backward_25[0]
    getitem_281: "f32[64, 256, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_737: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_280, mul_736);  getitem_280 = mul_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_168: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3])
    sub_236: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_406);  convolution_8 = unsqueeze_406 = None
    mul_738: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_737, sub_236)
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_738, [0, 2, 3]);  mul_738 = None
    mul_739: "f32[256]" = torch.ops.aten.mul.Tensor(sum_168, 3.0517578125e-05)
    unsqueeze_407: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_408: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_740: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, 3.0517578125e-05)
    mul_741: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_742: "f32[256]" = torch.ops.aten.mul.Tensor(mul_740, mul_741);  mul_740 = mul_741 = None
    unsqueeze_410: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_411: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_743: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_413: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_414: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_744: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_412);  sub_236 = unsqueeze_412 = None
    sub_238: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_737, mul_744);  mul_737 = mul_744 = None
    sub_239: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_409);  sub_238 = unsqueeze_409 = None
    mul_745: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_415);  sub_239 = unsqueeze_415 = None
    mul_746: "f32[256]" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_745, mul_61, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_745 = mul_61 = primals_73 = None
    getitem_283: "f32[8, 256, 64, 64]" = convolution_backward_26[0]
    getitem_284: "f32[256, 1, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_749: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_283, mul_748);  getitem_283 = mul_748 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3])
    sub_241: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_418);  convolution_7 = unsqueeze_418 = None
    mul_750: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(mul_749, sub_241)
    sum_171: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
    mul_751: "f32[256]" = torch.ops.aten.mul.Tensor(sum_170, 3.0517578125e-05)
    unsqueeze_419: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_420: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_752: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, 3.0517578125e-05)
    mul_753: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_754: "f32[256]" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
    unsqueeze_422: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_423: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_755: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_425: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_426: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_756: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_424);  sub_241 = unsqueeze_424 = None
    sub_243: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(mul_749, mul_756);  mul_749 = mul_756 = None
    sub_244: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_421);  sub_243 = unsqueeze_421 = None
    mul_757: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_427);  sub_244 = unsqueeze_427 = None
    mul_758: "f32[256]" = torch.ops.aten.mul.Tensor(sum_171, squeeze_22);  sum_171 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_757, add_34, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_757 = add_34 = primals_72 = None
    getitem_286: "f32[8, 64, 64, 64]" = convolution_backward_27[0]
    getitem_287: "f32[256, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_273: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(add_270, getitem_286);  add_270 = getitem_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_172: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
    sub_245: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_430);  convolution_6 = unsqueeze_430 = None
    mul_759: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(add_273, sub_245)
    sum_173: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_760: "f32[64]" = torch.ops.aten.mul.Tensor(sum_172, 3.0517578125e-05)
    unsqueeze_431: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_432: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_761: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, 3.0517578125e-05)
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_763: "f32[64]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    unsqueeze_434: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_435: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_764: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_437: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_438: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_765: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_436);  sub_245 = unsqueeze_436 = None
    sub_247: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(add_273, mul_765);  add_273 = mul_765 = None
    sub_248: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_433);  sub_247 = unsqueeze_433 = None
    mul_766: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_439);  sub_248 = unsqueeze_439 = None
    mul_767: "f32[64]" = torch.ops.aten.mul.Tensor(sum_173, squeeze_19);  sum_173 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_766, mul_46, primals_71, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = mul_46 = primals_71 = None
    getitem_289: "f32[8, 128, 64, 64]" = convolution_backward_28[0]
    getitem_290: "f32[64, 128, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_770: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(getitem_289, mul_769);  getitem_289 = mul_769 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_174: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3])
    sub_250: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_442);  convolution_5 = unsqueeze_442 = None
    mul_771: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(mul_770, sub_250)
    sum_175: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
    mul_772: "f32[128]" = torch.ops.aten.mul.Tensor(sum_174, 3.0517578125e-05)
    unsqueeze_443: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_444: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_773: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, 3.0517578125e-05)
    mul_774: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_775: "f32[128]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
    unsqueeze_446: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_447: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_776: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_449: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_450: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_777: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_448);  sub_250 = unsqueeze_448 = None
    sub_252: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(mul_770, mul_777);  mul_770 = mul_777 = None
    sub_253: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_445);  sub_252 = unsqueeze_445 = None
    mul_778: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_451);  sub_253 = unsqueeze_451 = None
    mul_779: "f32[128]" = torch.ops.aten.mul.Tensor(sum_175, squeeze_16);  sum_175 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_778, mul_38, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_778 = mul_38 = primals_70 = None
    getitem_292: "f32[8, 128, 128, 128]" = convolution_backward_29[0]
    getitem_293: "f32[128, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_782: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_292, mul_781);  getitem_292 = mul_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_176: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_782, [0, 2, 3])
    sub_255: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_454);  convolution_4 = unsqueeze_454 = None
    mul_783: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(mul_782, sub_255)
    sum_177: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_783, [0, 2, 3]);  mul_783 = None
    mul_784: "f32[128]" = torch.ops.aten.mul.Tensor(sum_176, 7.62939453125e-06)
    unsqueeze_455: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_456: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_785: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, 7.62939453125e-06)
    mul_786: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_787: "f32[128]" = torch.ops.aten.mul.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
    unsqueeze_458: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_459: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_788: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_461: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_462: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_789: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_460);  sub_255 = unsqueeze_460 = None
    sub_257: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(mul_782, mul_789);  mul_782 = mul_789 = None
    sub_258: "f32[8, 128, 128, 128]" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_457);  sub_257 = unsqueeze_457 = None
    mul_790: "f32[8, 128, 128, 128]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_463);  sub_258 = unsqueeze_463 = None
    mul_791: "f32[128]" = torch.ops.aten.mul.Tensor(sum_177, squeeze_13);  sum_177 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_790, add_19, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_790 = add_19 = primals_69 = None
    getitem_295: "f32[8, 32, 128, 128]" = convolution_backward_30[0]
    getitem_296: "f32[128, 32, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_178: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_295, [0, 2, 3])
    sub_259: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_466);  convolution_3 = unsqueeze_466 = None
    mul_792: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_295, sub_259)
    sum_179: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 2, 3]);  mul_792 = None
    mul_793: "f32[32]" = torch.ops.aten.mul.Tensor(sum_178, 7.62939453125e-06)
    unsqueeze_467: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_468: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_794: "f32[32]" = torch.ops.aten.mul.Tensor(sum_179, 7.62939453125e-06)
    mul_795: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_796: "f32[32]" = torch.ops.aten.mul.Tensor(mul_794, mul_795);  mul_794 = mul_795 = None
    unsqueeze_470: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_471: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_797: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_473: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_474: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_798: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_472);  sub_259 = unsqueeze_472 = None
    sub_261: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(getitem_295, mul_798);  getitem_295 = mul_798 = None
    sub_262: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_469);  sub_261 = unsqueeze_469 = None
    mul_799: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_475);  sub_262 = unsqueeze_475 = None
    mul_800: "f32[32]" = torch.ops.aten.mul.Tensor(sum_179, squeeze_10);  sum_179 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_799, mul_23, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_799 = mul_23 = primals_68 = None
    getitem_298: "f32[8, 64, 128, 128]" = convolution_backward_31[0]
    getitem_299: "f32[32, 64, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_803: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_298, mul_802);  getitem_298 = mul_802 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_180: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 2, 3])
    sub_264: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_478);  convolution_2 = unsqueeze_478 = None
    mul_804: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_803, sub_264)
    sum_181: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_804, [0, 2, 3]);  mul_804 = None
    mul_805: "f32[64]" = torch.ops.aten.mul.Tensor(sum_180, 7.62939453125e-06)
    unsqueeze_479: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_805, 0);  mul_805 = None
    unsqueeze_480: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_806: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, 7.62939453125e-06)
    mul_807: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_808: "f32[64]" = torch.ops.aten.mul.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    unsqueeze_482: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_483: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_809: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_485: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_486: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_810: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_484);  sub_264 = unsqueeze_484 = None
    sub_266: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_803, mul_810);  mul_803 = mul_810 = None
    sub_267: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_481);  sub_266 = unsqueeze_481 = None
    mul_811: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_487);  sub_267 = unsqueeze_487 = None
    mul_812: "f32[64]" = torch.ops.aten.mul.Tensor(sum_181, squeeze_7);  sum_181 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_811, mul_15, primals_67, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_811 = mul_15 = primals_67 = None
    getitem_301: "f32[8, 64, 128, 128]" = convolution_backward_32[0]
    getitem_302: "f32[64, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_815: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_301, mul_814);  getitem_301 = mul_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_182: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3])
    sub_269: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_490);  convolution_1 = unsqueeze_490 = None
    mul_816: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(mul_815, sub_269)
    sum_183: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None
    mul_817: "f32[64]" = torch.ops.aten.mul.Tensor(sum_182, 7.62939453125e-06)
    unsqueeze_491: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_492: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_818: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, 7.62939453125e-06)
    mul_819: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_820: "f32[64]" = torch.ops.aten.mul.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
    unsqueeze_494: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_495: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_821: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_497: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_498: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_822: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_496);  sub_269 = unsqueeze_496 = None
    sub_271: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(mul_815, mul_822);  mul_815 = mul_822 = None
    sub_272: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_493);  sub_271 = unsqueeze_493 = None
    mul_823: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_499);  sub_272 = unsqueeze_499 = None
    mul_824: "f32[64]" = torch.ops.aten.mul.Tensor(sum_183, squeeze_4);  sum_183 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_823, mul_7, primals_66, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_823 = mul_7 = primals_66 = None
    getitem_304: "f32[8, 16, 128, 128]" = convolution_backward_33[0]
    getitem_305: "f32[64, 16, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    mul_827: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(getitem_304, mul_826);  getitem_304 = mul_826 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_184: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3])
    sub_274: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_502);  convolution = unsqueeze_502 = None
    mul_828: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(mul_827, sub_274)
    sum_185: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2, 3]);  mul_828 = None
    mul_829: "f32[16]" = torch.ops.aten.mul.Tensor(sum_184, 7.62939453125e-06)
    unsqueeze_503: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_504: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_830: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, 7.62939453125e-06)
    mul_831: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_832: "f32[16]" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
    unsqueeze_506: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_507: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_833: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_509: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
    unsqueeze_510: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_834: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_508);  sub_274 = unsqueeze_508 = None
    sub_276: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(mul_827, mul_834);  mul_827 = mul_834 = None
    sub_277: "f32[8, 16, 128, 128]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_505);  sub_276 = unsqueeze_505 = None
    mul_835: "f32[8, 16, 128, 128]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_511);  sub_277 = unsqueeze_511 = None
    mul_836: "f32[16]" = torch.ops.aten.mul.Tensor(sum_185, squeeze_1);  sum_185 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_835, primals_312, primals_65, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_835 = primals_312 = primals_65 = None
    getitem_308: "f32[16, 3, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    return [mul_836, sum_184, mul_824, sum_182, mul_812, sum_180, mul_800, sum_178, mul_791, sum_176, mul_779, sum_174, mul_767, sum_172, mul_758, sum_170, mul_746, sum_168, mul_734, sum_166, mul_725, sum_164, mul_713, sum_162, mul_701, sum_160, mul_692, sum_158, mul_680, sum_156, mul_668, sum_154, mul_659, sum_152, mul_606, sum_122, mul_594, sum_120, mul_582, sum_118, mul_570, sum_116, mul_558, sum_114, mul_549, sum_112, mul_462, sum_58, mul_450, sum_56, mul_438, sum_54, mul_426, sum_52, mul_414, sum_50, mul_405, sum_48, mul_335, sum_6, mul_323, sum_4, mul_311, sum_2, getitem_308, getitem_305, getitem_302, getitem_299, getitem_296, getitem_293, getitem_290, getitem_287, getitem_284, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_266, getitem_263, getitem_260, getitem_257, sum_150, sum_151, permute_264, view_259, permute_258, view_253, sum_144, sum_145, permute_254, view_250, permute_249, view_247, sum_138, sum_139, permute_245, view_244, permute_239, view_238, sum_132, sum_133, permute_235, view_235, permute_230, view_232, sum_126, sum_127, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, sum_110, sum_111, permute_217, view_223, permute_211, view_217, sum_104, sum_105, permute_207, view_214, permute_202, view_211, sum_98, sum_99, permute_198, view_208, permute_192, view_202, sum_92, sum_93, permute_188, view_199, permute_183, view_196, sum_86, sum_87, permute_179, view_193, permute_173, view_187, sum_80, sum_81, permute_169, view_184, permute_164, view_181, sum_74, sum_75, permute_160, view_178, permute_154, view_172, sum_68, sum_69, permute_150, view_169, permute_145, view_166, sum_62, sum_63, getitem_209, getitem_206, getitem_203, getitem_200, getitem_197, getitem_194, getitem_191, sum_46, sum_47, permute_132, view_157, permute_126, view_151, sum_40, sum_41, permute_122, view_148, permute_117, view_145, sum_34, sum_35, permute_113, view_142, permute_107, view_136, sum_28, sum_29, permute_103, view_133, permute_98, view_130, sum_22, sum_23, permute_94, view_127, permute_88, view_121, sum_16, sum_17, permute_84, view_118, permute_79, view_115, sum_10, sum_11, getitem_176, getitem_173, getitem_170, permute_70, view_109, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    