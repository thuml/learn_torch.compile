from __future__ import annotations



def forward(self, primals_4: "f32[32, 3, 7, 7]", primals_5: "f32[32]", primals_7: "f32[192, 32, 4, 4]", primals_9: "f32[192]", primals_11: "f32[192]", primals_13: "f32[384, 192, 1, 1]", primals_14: "f32[384, 48, 3, 3]", primals_15: "f32[192, 384, 1, 1]", primals_16: "f32[192]", primals_18: "f32[384, 192, 1, 1]", primals_19: "f32[384, 48, 3, 3]", primals_20: "f32[192, 384, 1, 1]", primals_21: "f32[192]", primals_23: "f32[384, 192, 1, 1]", primals_24: "f32[384, 48, 3, 3]", primals_25: "f32[192, 384, 1, 1]", primals_26: "f32[192]", primals_28: "f32[384, 192, 1, 1]", primals_29: "f32[384, 48, 3, 3]", primals_30: "f32[192, 384, 1, 1]", primals_31: "f32[192]", primals_33: "f32[384, 192, 1, 1]", primals_34: "f32[384, 48, 3, 3]", primals_35: "f32[192, 384, 1, 1]", primals_36: "f32[192]", primals_38: "f32[384, 192, 1, 1]", primals_39: "f32[384, 48, 3, 3]", primals_40: "f32[192, 384, 1, 1]", primals_41: "f32[192]", primals_43: "f32[384, 192, 1, 1]", primals_44: "f32[384, 48, 3, 3]", primals_45: "f32[192, 384, 1, 1]", primals_46: "f32[384, 192, 2, 2]", primals_48: "f32[384]", primals_50: "f32[384]", primals_52: "f32[1152, 384, 1, 1]", primals_53: "f32[384, 384, 1, 1]", primals_54: "f32[384]", primals_56: "f32[1536, 384, 1, 1]", primals_57: "f32[384, 1536, 1, 1]", primals_58: "f32[384]", primals_60: "f32[1152, 384, 1, 1]", primals_61: "f32[384, 384, 1, 1]", primals_62: "f32[384]", primals_64: "f32[1536, 384, 1, 1]", primals_65: "f32[384, 1536, 1, 1]", primals_66: "f32[384]", primals_68: "f32[1152, 384, 1, 1]", primals_69: "f32[384, 384, 1, 1]", primals_70: "f32[384]", primals_72: "f32[1536, 384, 1, 1]", primals_73: "f32[384, 1536, 1, 1]", primals_74: "f32[384]", primals_76: "f32[1152, 384, 1, 1]", primals_77: "f32[384, 384, 1, 1]", primals_78: "f32[384]", primals_80: "f32[1536, 384, 1, 1]", primals_81: "f32[384, 1536, 1, 1]", primals_82: "f32[768, 384, 2, 2]", primals_84: "f32[768]", primals_86: "f32[768]", primals_88: "f32[2304, 768, 1, 1]", primals_89: "f32[768, 768, 1, 1]", primals_90: "f32[768]", primals_92: "f32[3072, 768, 1, 1]", primals_93: "f32[768, 3072, 1, 1]", primals_94: "f32[768]", primals_96: "f32[2304, 768, 1, 1]", primals_97: "f32[768, 768, 1, 1]", primals_98: "f32[768]", primals_100: "f32[3072, 768, 1, 1]", primals_101: "f32[768, 3072, 1, 1]", primals_102: "f32[768]", primals_104: "f32[2304, 768, 1, 1]", primals_105: "f32[768, 768, 1, 1]", primals_106: "f32[768]", primals_108: "f32[3072, 768, 1, 1]", primals_109: "f32[768, 3072, 1, 1]", primals_110: "f32[768]", primals_112: "f32[2304, 768, 1, 1]", primals_113: "f32[768, 768, 1, 1]", primals_114: "f32[768]", primals_116: "f32[3072, 768, 1, 1]", primals_117: "f32[768, 3072, 1, 1]", primals_118: "f32[768]", primals_206: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 192, 28, 28]", squeeze_4: "f32[192]", clone: "f32[8, 192, 28, 28]", squeeze_7: "f32[192]", add_15: "f32[8, 192, 28, 28]", convolution_2: "f32[8, 384, 28, 28]", clone_1: "f32[8, 384, 28, 28]", convolution_3: "f32[8, 384, 28, 28]", mul_26: "f32[8, 384, 28, 28]", convolution_4: "f32[8, 192, 28, 28]", squeeze_10: "f32[192]", add_23: "f32[8, 192, 28, 28]", convolution_5: "f32[8, 384, 28, 28]", clone_3: "f32[8, 384, 28, 28]", convolution_6: "f32[8, 384, 28, 28]", mul_39: "f32[8, 384, 28, 28]", convolution_7: "f32[8, 192, 28, 28]", squeeze_13: "f32[192]", add_31: "f32[8, 192, 28, 28]", convolution_8: "f32[8, 384, 28, 28]", clone_5: "f32[8, 384, 28, 28]", convolution_9: "f32[8, 384, 28, 28]", mul_52: "f32[8, 384, 28, 28]", convolution_10: "f32[8, 192, 28, 28]", squeeze_16: "f32[192]", add_39: "f32[8, 192, 28, 28]", convolution_11: "f32[8, 384, 28, 28]", clone_7: "f32[8, 384, 28, 28]", convolution_12: "f32[8, 384, 28, 28]", mul_65: "f32[8, 384, 28, 28]", convolution_13: "f32[8, 192, 28, 28]", squeeze_19: "f32[192]", add_47: "f32[8, 192, 28, 28]", convolution_14: "f32[8, 384, 28, 28]", clone_9: "f32[8, 384, 28, 28]", convolution_15: "f32[8, 384, 28, 28]", mul_78: "f32[8, 384, 28, 28]", convolution_16: "f32[8, 192, 28, 28]", squeeze_22: "f32[192]", add_55: "f32[8, 192, 28, 28]", convolution_17: "f32[8, 384, 28, 28]", clone_11: "f32[8, 384, 28, 28]", convolution_18: "f32[8, 384, 28, 28]", mul_91: "f32[8, 384, 28, 28]", convolution_19: "f32[8, 192, 28, 28]", squeeze_25: "f32[192]", add_63: "f32[8, 192, 28, 28]", convolution_20: "f32[8, 384, 28, 28]", clone_13: "f32[8, 384, 28, 28]", convolution_21: "f32[8, 384, 28, 28]", mul_104: "f32[8, 384, 28, 28]", add_66: "f32[8, 192, 28, 28]", convolution_23: "f32[8, 384, 14, 14]", squeeze_28: "f32[384]", clone_15: "f32[8, 384, 14, 14]", squeeze_31: "f32[384]", add_77: "f32[8, 384, 14, 14]", view_7: "f32[8, 384, 14, 14]", convolution_25: "f32[8, 384, 14, 14]", squeeze_34: "f32[384]", add_83: "f32[8, 384, 14, 14]", convolution_26: "f32[8, 1536, 14, 14]", clone_22: "f32[8, 1536, 14, 14]", convolution_27: "f32[8, 384, 14, 14]", squeeze_37: "f32[384]", add_90: "f32[8, 384, 14, 14]", view_15: "f32[8, 384, 14, 14]", convolution_29: "f32[8, 384, 14, 14]", squeeze_40: "f32[384]", add_96: "f32[8, 384, 14, 14]", convolution_30: "f32[8, 1536, 14, 14]", clone_30: "f32[8, 1536, 14, 14]", convolution_31: "f32[8, 384, 14, 14]", squeeze_43: "f32[384]", add_103: "f32[8, 384, 14, 14]", view_23: "f32[8, 384, 14, 14]", convolution_33: "f32[8, 384, 14, 14]", squeeze_46: "f32[384]", add_109: "f32[8, 384, 14, 14]", convolution_34: "f32[8, 1536, 14, 14]", clone_38: "f32[8, 1536, 14, 14]", convolution_35: "f32[8, 384, 14, 14]", squeeze_49: "f32[384]", add_116: "f32[8, 384, 14, 14]", view_31: "f32[8, 384, 14, 14]", convolution_37: "f32[8, 384, 14, 14]", squeeze_52: "f32[384]", add_122: "f32[8, 384, 14, 14]", convolution_38: "f32[8, 1536, 14, 14]", clone_46: "f32[8, 1536, 14, 14]", add_124: "f32[8, 384, 14, 14]", convolution_40: "f32[8, 768, 7, 7]", squeeze_55: "f32[768]", clone_48: "f32[8, 768, 7, 7]", squeeze_58: "f32[768]", add_135: "f32[8, 768, 7, 7]", view_39: "f32[8, 768, 7, 7]", convolution_42: "f32[8, 768, 7, 7]", squeeze_61: "f32[768]", add_141: "f32[8, 768, 7, 7]", convolution_43: "f32[8, 3072, 7, 7]", clone_55: "f32[8, 3072, 7, 7]", convolution_44: "f32[8, 768, 7, 7]", squeeze_64: "f32[768]", add_148: "f32[8, 768, 7, 7]", view_47: "f32[8, 768, 7, 7]", convolution_46: "f32[8, 768, 7, 7]", squeeze_67: "f32[768]", add_154: "f32[8, 768, 7, 7]", convolution_47: "f32[8, 3072, 7, 7]", clone_63: "f32[8, 3072, 7, 7]", convolution_48: "f32[8, 768, 7, 7]", squeeze_70: "f32[768]", add_161: "f32[8, 768, 7, 7]", view_55: "f32[8, 768, 7, 7]", convolution_50: "f32[8, 768, 7, 7]", squeeze_73: "f32[768]", add_167: "f32[8, 768, 7, 7]", convolution_51: "f32[8, 3072, 7, 7]", clone_71: "f32[8, 3072, 7, 7]", convolution_52: "f32[8, 768, 7, 7]", squeeze_76: "f32[768]", add_174: "f32[8, 768, 7, 7]", view_63: "f32[8, 768, 7, 7]", convolution_54: "f32[8, 768, 7, 7]", squeeze_79: "f32[768]", add_180: "f32[8, 768, 7, 7]", convolution_55: "f32[8, 3072, 7, 7]", clone_79: "f32[8, 3072, 7, 7]", convolution_56: "f32[8, 768, 7, 7]", squeeze_82: "f32[768]", clone_81: "f32[8, 768]", permute_25: "f32[1000, 768]", unsqueeze_114: "f32[1, 768, 1, 1]", unsqueeze_126: "f32[1, 768, 1, 1]", permute_30: "f32[48, 49, 49]", permute_31: "f32[48, 128, 49]", alias_9: "f32[8, 6, 49, 49]", permute_32: "f32[48, 128, 49]", permute_33: "f32[48, 49, 128]", unsqueeze_138: "f32[1, 768, 1, 1]", unsqueeze_150: "f32[1, 768, 1, 1]", permute_37: "f32[48, 49, 49]", permute_38: "f32[48, 128, 49]", alias_10: "f32[8, 6, 49, 49]", permute_39: "f32[48, 128, 49]", permute_40: "f32[48, 49, 128]", unsqueeze_162: "f32[1, 768, 1, 1]", unsqueeze_174: "f32[1, 768, 1, 1]", permute_44: "f32[48, 49, 49]", permute_45: "f32[48, 128, 49]", alias_11: "f32[8, 6, 49, 49]", permute_46: "f32[48, 128, 49]", permute_47: "f32[48, 49, 128]", unsqueeze_186: "f32[1, 768, 1, 1]", unsqueeze_198: "f32[1, 768, 1, 1]", permute_51: "f32[48, 49, 49]", permute_52: "f32[48, 128, 49]", alias_12: "f32[8, 6, 49, 49]", permute_53: "f32[48, 128, 49]", permute_54: "f32[48, 49, 128]", unsqueeze_210: "f32[1, 768, 1, 1]", unsqueeze_222: "f32[1, 768, 1, 1]", unsqueeze_234: "f32[1, 384, 1, 1]", permute_58: "f32[48, 196, 196]", permute_59: "f32[48, 64, 196]", alias_13: "f32[8, 6, 196, 196]", permute_60: "f32[48, 64, 196]", permute_61: "f32[48, 196, 64]", unsqueeze_246: "f32[1, 384, 1, 1]", unsqueeze_258: "f32[1, 384, 1, 1]", permute_65: "f32[48, 196, 196]", permute_66: "f32[48, 64, 196]", alias_14: "f32[8, 6, 196, 196]", permute_67: "f32[48, 64, 196]", permute_68: "f32[48, 196, 64]", unsqueeze_270: "f32[1, 384, 1, 1]", unsqueeze_282: "f32[1, 384, 1, 1]", permute_72: "f32[48, 196, 196]", permute_73: "f32[48, 64, 196]", alias_15: "f32[8, 6, 196, 196]", permute_74: "f32[48, 64, 196]", permute_75: "f32[48, 196, 64]", unsqueeze_294: "f32[1, 384, 1, 1]", unsqueeze_306: "f32[1, 384, 1, 1]", permute_79: "f32[48, 196, 196]", permute_80: "f32[48, 64, 196]", alias_16: "f32[8, 6, 196, 196]", permute_81: "f32[48, 64, 196]", permute_82: "f32[48, 196, 64]", unsqueeze_318: "f32[1, 384, 1, 1]", unsqueeze_330: "f32[1, 384, 1, 1]", unsqueeze_342: "f32[1, 192, 1, 1]", unsqueeze_354: "f32[1, 192, 1, 1]", unsqueeze_366: "f32[1, 192, 1, 1]", unsqueeze_378: "f32[1, 192, 1, 1]", unsqueeze_390: "f32[1, 192, 1, 1]", unsqueeze_402: "f32[1, 192, 1, 1]", unsqueeze_414: "f32[1, 192, 1, 1]", unsqueeze_426: "f32[1, 192, 1, 1]", unsqueeze_438: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_22: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
    erf: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
    add_16: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_25: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, 0.7071067811865476)
    erf_1: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
    add_17: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_18: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(clone, convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_35: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
    erf_2: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_24: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_38: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, 0.7071067811865476)
    erf_3: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_25: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_26: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_18, convolution_7);  convolution_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_48: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, 0.7071067811865476)
    erf_4: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_32: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_51: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
    erf_5: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_33: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_34: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_26, convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_61: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, 0.7071067811865476)
    erf_6: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_40: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_64: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
    erf_7: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_41: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_42: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_34, convolution_13);  convolution_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_74: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, 0.7071067811865476)
    erf_8: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_48: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_77: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, 0.7071067811865476)
    erf_9: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_49: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_50: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_42, convolution_16);  convolution_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_87: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, 0.7071067811865476)
    erf_10: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_56: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_90: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, 0.7071067811865476)
    erf_11: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_57: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_58: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_50, convolution_19);  convolution_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_100: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, 0.7071067811865476)
    erf_12: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_64: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_103: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, 0.7071067811865476)
    erf_13: "f32[8, 384, 28, 28]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_65: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_78: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(clone_15, convolution_25);  convolution_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_128: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, 0.7071067811865476)
    erf_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_84: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_85: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_78, convolution_27);  convolution_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_91: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_85, convolution_29);  convolution_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_146: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, 0.7071067811865476)
    erf_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_97: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_98: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_91, convolution_31);  convolution_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_104: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_98, convolution_33);  convolution_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_164: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
    erf_16: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_110: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_111: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_104, convolution_35);  convolution_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_117: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_111, convolution_37);  convolution_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_182: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, 0.7071067811865476)
    erf_17: "f32[8, 1536, 14, 14]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_123: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_136: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(clone_48, convolution_42);  convolution_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_207: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, 0.7071067811865476)
    erf_18: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_142: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_143: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_136, convolution_44);  convolution_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_149: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_143, convolution_46);  convolution_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_225: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, 0.7071067811865476)
    erf_19: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_225);  mul_225 = None
    add_155: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_156: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_149, convolution_48);  convolution_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_162: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_156, convolution_50);  convolution_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_243: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, 0.7071067811865476)
    erf_20: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
    add_168: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_169: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_162, convolution_52);  convolution_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_175: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_169, convolution_54);  convolution_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_261: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, 0.7071067811865476)
    erf_21: "f32[8, 3072, 7, 7]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
    add_181: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_182: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_175, convolution_56);  convolution_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:433, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
    permute_26: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_26, clone_81);  permute_26 = clone_81 = None
    permute_27: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_9: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_65: "f32[1000]" = torch.ops.aten.reshape.default(sum_9, [1000]);  sum_9 = None
    permute_28: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_66: "f32[8, 768, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 768, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_32: "f32[8, 768, 7, 7]" = torch.ops.aten.expand.default(view_66, [8, 768, 7, 7]);  view_66 = None
    div_8: "f32[8, 768, 7, 7]" = torch.ops.aten.div.Scalar(expand_32, 49);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:427, code: x = self.norm(x)
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(div_8, [0, 2, 3])
    sub_36: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_182, unsqueeze_114);  add_182 = unsqueeze_114 = None
    mul_270: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(div_8, sub_36)
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 2, 3]);  mul_270 = None
    mul_271: "f32[768]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_115: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
    unsqueeze_116: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_115, 2);  unsqueeze_115 = None
    unsqueeze_117: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, 3);  unsqueeze_116 = None
    mul_272: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_273: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_274: "f32[768]" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    unsqueeze_118: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
    unsqueeze_119: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, 2);  unsqueeze_118 = None
    unsqueeze_120: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 3);  unsqueeze_119 = None
    mul_275: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_118);  primals_118 = None
    unsqueeze_121: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
    unsqueeze_122: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 2);  unsqueeze_121 = None
    unsqueeze_123: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 3);  unsqueeze_122 = None
    mul_276: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_120);  sub_36 = unsqueeze_120 = None
    sub_38: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(div_8, mul_276);  div_8 = mul_276 = None
    sub_39: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_38, unsqueeze_117);  sub_38 = unsqueeze_117 = None
    mul_277: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_123);  sub_39 = unsqueeze_123 = None
    mul_278: "f32[768]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_82);  sum_11 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_277, clone_79, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_79 = primals_117 = None
    getitem_80: "f32[8, 3072, 7, 7]" = convolution_backward[0]
    getitem_81: "f32[768, 3072, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_280: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_181, 0.5);  add_181 = None
    mul_281: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, convolution_55)
    mul_282: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_281, -0.5);  mul_281 = None
    exp_8: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_282);  mul_282 = None
    mul_283: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_284: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_55, mul_283);  convolution_55 = mul_283 = None
    add_189: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_280, mul_284);  mul_280 = mul_284 = None
    mul_285: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_80, add_189);  getitem_80 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_285, add_180, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_285 = add_180 = primals_116 = None
    getitem_83: "f32[8, 768, 7, 7]" = convolution_backward_1[0]
    getitem_84: "f32[3072, 768, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_83, [0, 2, 3])
    sub_40: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_175, unsqueeze_126);  add_175 = unsqueeze_126 = None
    mul_286: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_83, sub_40)
    sum_13: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 2, 3]);  mul_286 = None
    mul_287: "f32[768]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_127: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
    unsqueeze_128: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_288: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_289: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_290: "f32[768]" = torch.ops.aten.mul.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    unsqueeze_130: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_131: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_291: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_114);  primals_114 = None
    unsqueeze_133: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_134: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    mul_292: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_132);  sub_40 = unsqueeze_132 = None
    sub_42: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_83, mul_292);  getitem_83 = mul_292 = None
    sub_43: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_129);  sub_42 = unsqueeze_129 = None
    mul_293: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_135);  sub_43 = unsqueeze_135 = None
    mul_294: "f32[768]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_79);  sum_13 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_190: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(mul_277, mul_293);  mul_277 = mul_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(add_190, view_63, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_63 = primals_113 = None
    getitem_86: "f32[8, 768, 7, 7]" = convolution_backward_2[0]
    getitem_87: "f32[768, 768, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_67: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(getitem_86, [8, 6, 128, 49]);  getitem_86 = None
    permute_29: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_67, [0, 1, 3, 2]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_68: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(permute_29, [48, 49, 128]);  permute_29 = None
    bmm_16: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_30, view_68);  permute_30 = None
    bmm_17: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_68, permute_31);  view_68 = permute_31 = None
    view_69: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_16, [8, 6, 49, 128]);  bmm_16 = None
    view_70: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_17, [8, 6, 49, 49]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_295: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_70, alias_9);  view_70 = None
    sum_14: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [-1], True)
    mul_296: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_9, sum_14);  alias_9 = sum_14 = None
    sub_44: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_297: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_44, 0.08838834764831845);  sub_44 = None
    view_71: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(mul_297, [48, 49, 49]);  mul_297 = None
    bmm_18: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_32, view_71);  permute_32 = None
    bmm_19: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_71, permute_33);  view_71 = permute_33 = None
    view_72: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(bmm_18, [8, 6, 128, 49]);  bmm_18 = None
    view_73: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_19, [8, 6, 49, 128]);  bmm_19 = None
    permute_34: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_72, [0, 1, 3, 2]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_73, permute_34, view_69]);  view_73 = permute_34 = view_69 = None
    view_74: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.reshape.default(cat, [3, 8, 6, 49, 128]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_35: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_74, [1, 0, 2, 4, 3]);  view_74 = None
    clone_82: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    view_75: "f32[8, 2304, 7, 7]" = torch.ops.aten.reshape.default(clone_82, [8, 2304, 7, 7]);  clone_82 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_75, add_174, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_75 = add_174 = primals_112 = None
    getitem_89: "f32[8, 768, 7, 7]" = convolution_backward_3[0]
    getitem_90: "f32[2304, 768, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_89, [0, 2, 3])
    sub_45: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_169, unsqueeze_138);  add_169 = unsqueeze_138 = None
    mul_298: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_89, sub_45)
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3]);  mul_298 = None
    mul_299: "f32[768]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    unsqueeze_139: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_140: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_300: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    mul_301: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_302: "f32[768]" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_142: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_143: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_303: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_110);  primals_110 = None
    unsqueeze_145: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
    unsqueeze_146: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    mul_304: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_144);  sub_45 = unsqueeze_144 = None
    sub_47: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_89, mul_304);  getitem_89 = mul_304 = None
    sub_48: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_47, unsqueeze_141);  sub_47 = unsqueeze_141 = None
    mul_305: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_147);  sub_48 = unsqueeze_147 = None
    mul_306: "f32[768]" = torch.ops.aten.mul.Tensor(sum_16, squeeze_76);  sum_16 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_191: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_190, mul_305);  add_190 = mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(add_191, clone_71, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_71 = primals_109 = None
    getitem_92: "f32[8, 3072, 7, 7]" = convolution_backward_4[0]
    getitem_93: "f32[768, 3072, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_308: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
    mul_309: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, convolution_51)
    mul_310: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_309, -0.5);  mul_309 = None
    exp_9: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_310);  mul_310 = None
    mul_311: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_312: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_51, mul_311);  convolution_51 = mul_311 = None
    add_193: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_308, mul_312);  mul_308 = mul_312 = None
    mul_313: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_92, add_193);  getitem_92 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_313, add_167, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_313 = add_167 = primals_108 = None
    getitem_95: "f32[8, 768, 7, 7]" = convolution_backward_5[0]
    getitem_96: "f32[3072, 768, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_95, [0, 2, 3])
    sub_49: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_162, unsqueeze_150);  add_162 = unsqueeze_150 = None
    mul_314: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_95, sub_49)
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_315: "f32[768]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    unsqueeze_151: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_152: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_316: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    mul_317: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_318: "f32[768]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_154: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_155: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_319: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_106);  primals_106 = None
    unsqueeze_157: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_158: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    mul_320: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_156);  sub_49 = unsqueeze_156 = None
    sub_51: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_95, mul_320);  getitem_95 = mul_320 = None
    sub_52: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_153);  sub_51 = unsqueeze_153 = None
    mul_321: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_159);  sub_52 = unsqueeze_159 = None
    mul_322: "f32[768]" = torch.ops.aten.mul.Tensor(sum_18, squeeze_73);  sum_18 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_194: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_191, mul_321);  add_191 = mul_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_194, view_55, primals_105, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_55 = primals_105 = None
    getitem_98: "f32[8, 768, 7, 7]" = convolution_backward_6[0]
    getitem_99: "f32[768, 768, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_76: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(getitem_98, [8, 6, 128, 49]);  getitem_98 = None
    permute_36: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_76, [0, 1, 3, 2]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_77: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(permute_36, [48, 49, 128]);  permute_36 = None
    bmm_20: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_37, view_77);  permute_37 = None
    bmm_21: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_77, permute_38);  view_77 = permute_38 = None
    view_78: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_20, [8, 6, 49, 128]);  bmm_20 = None
    view_79: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_21, [8, 6, 49, 49]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_323: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_79, alias_10);  view_79 = None
    sum_19: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
    mul_324: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_10, sum_19);  alias_10 = sum_19 = None
    sub_53: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_325: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_53, 0.08838834764831845);  sub_53 = None
    view_80: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(mul_325, [48, 49, 49]);  mul_325 = None
    bmm_22: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_39, view_80);  permute_39 = None
    bmm_23: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_80, permute_40);  view_80 = permute_40 = None
    view_81: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(bmm_22, [8, 6, 128, 49]);  bmm_22 = None
    view_82: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_23, [8, 6, 49, 128]);  bmm_23 = None
    permute_41: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_81, [0, 1, 3, 2]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_1: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_82, permute_41, view_78]);  view_82 = permute_41 = view_78 = None
    view_83: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.reshape.default(cat_1, [3, 8, 6, 49, 128]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_42: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_83, [1, 0, 2, 4, 3]);  view_83 = None
    clone_83: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_42, memory_format = torch.contiguous_format);  permute_42 = None
    view_84: "f32[8, 2304, 7, 7]" = torch.ops.aten.reshape.default(clone_83, [8, 2304, 7, 7]);  clone_83 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_84, add_161, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_84 = add_161 = primals_104 = None
    getitem_101: "f32[8, 768, 7, 7]" = convolution_backward_7[0]
    getitem_102: "f32[2304, 768, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_101, [0, 2, 3])
    sub_54: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_156, unsqueeze_162);  add_156 = unsqueeze_162 = None
    mul_326: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, sub_54)
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 2, 3]);  mul_326 = None
    mul_327: "f32[768]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_163: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_164: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_328: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_329: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_330: "f32[768]" = torch.ops.aten.mul.Tensor(mul_328, mul_329);  mul_328 = mul_329 = None
    unsqueeze_166: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_330, 0);  mul_330 = None
    unsqueeze_167: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_331: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_102);  primals_102 = None
    unsqueeze_169: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
    unsqueeze_170: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    mul_332: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_168);  sub_54 = unsqueeze_168 = None
    sub_56: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_101, mul_332);  getitem_101 = mul_332 = None
    sub_57: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_165);  sub_56 = unsqueeze_165 = None
    mul_333: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_171);  sub_57 = unsqueeze_171 = None
    mul_334: "f32[768]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_70);  sum_21 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_195: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_194, mul_333);  add_194 = mul_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(add_195, clone_63, primals_101, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_63 = primals_101 = None
    getitem_104: "f32[8, 3072, 7, 7]" = convolution_backward_8[0]
    getitem_105: "f32[768, 3072, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_336: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_155, 0.5);  add_155 = None
    mul_337: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, convolution_47)
    mul_338: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_337, -0.5);  mul_337 = None
    exp_10: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_338);  mul_338 = None
    mul_339: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_340: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_47, mul_339);  convolution_47 = mul_339 = None
    add_197: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_336, mul_340);  mul_336 = mul_340 = None
    mul_341: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_104, add_197);  getitem_104 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_341, add_154, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_341 = add_154 = primals_100 = None
    getitem_107: "f32[8, 768, 7, 7]" = convolution_backward_9[0]
    getitem_108: "f32[3072, 768, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_107, [0, 2, 3])
    sub_58: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_149, unsqueeze_174);  add_149 = unsqueeze_174 = None
    mul_342: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_107, sub_58)
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 2, 3]);  mul_342 = None
    mul_343: "f32[768]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_175: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_343, 0);  mul_343 = None
    unsqueeze_176: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 2);  unsqueeze_175 = None
    unsqueeze_177: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 3);  unsqueeze_176 = None
    mul_344: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_345: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_346: "f32[768]" = torch.ops.aten.mul.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
    unsqueeze_178: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_179: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_347: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_98);  primals_98 = None
    unsqueeze_181: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_182: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    mul_348: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_180);  sub_58 = unsqueeze_180 = None
    sub_60: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_107, mul_348);  getitem_107 = mul_348 = None
    sub_61: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_60, unsqueeze_177);  sub_60 = unsqueeze_177 = None
    mul_349: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_183);  sub_61 = unsqueeze_183 = None
    mul_350: "f32[768]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_67);  sum_23 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_198: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_195, mul_349);  add_195 = mul_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_198, view_47, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_47 = primals_97 = None
    getitem_110: "f32[8, 768, 7, 7]" = convolution_backward_10[0]
    getitem_111: "f32[768, 768, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_85: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(getitem_110, [8, 6, 128, 49]);  getitem_110 = None
    permute_43: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_85, [0, 1, 3, 2]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_86: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(permute_43, [48, 49, 128]);  permute_43 = None
    bmm_24: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_44, view_86);  permute_44 = None
    bmm_25: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_86, permute_45);  view_86 = permute_45 = None
    view_87: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_24, [8, 6, 49, 128]);  bmm_24 = None
    view_88: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_25, [8, 6, 49, 49]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_351: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_88, alias_11);  view_88 = None
    sum_24: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
    mul_352: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_11, sum_24);  alias_11 = sum_24 = None
    sub_62: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_351, mul_352);  mul_351 = mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_353: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_62, 0.08838834764831845);  sub_62 = None
    view_89: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(mul_353, [48, 49, 49]);  mul_353 = None
    bmm_26: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_46, view_89);  permute_46 = None
    bmm_27: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_89, permute_47);  view_89 = permute_47 = None
    view_90: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(bmm_26, [8, 6, 128, 49]);  bmm_26 = None
    view_91: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_27, [8, 6, 49, 128]);  bmm_27 = None
    permute_48: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_90, [0, 1, 3, 2]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_2: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_91, permute_48, view_87]);  view_91 = permute_48 = view_87 = None
    view_92: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.reshape.default(cat_2, [3, 8, 6, 49, 128]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_49: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_92, [1, 0, 2, 4, 3]);  view_92 = None
    clone_84: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    view_93: "f32[8, 2304, 7, 7]" = torch.ops.aten.reshape.default(clone_84, [8, 2304, 7, 7]);  clone_84 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(view_93, add_148, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_93 = add_148 = primals_96 = None
    getitem_113: "f32[8, 768, 7, 7]" = convolution_backward_11[0]
    getitem_114: "f32[2304, 768, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
    sub_63: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_143, unsqueeze_186);  add_143 = unsqueeze_186 = None
    mul_354: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_63)
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
    mul_355: "f32[768]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_187: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_188: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_356: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_357: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_358: "f32[768]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
    unsqueeze_190: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
    unsqueeze_191: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_359: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_94);  primals_94 = None
    unsqueeze_193: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_194: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_360: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_192);  sub_63 = unsqueeze_192 = None
    sub_65: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_360);  getitem_113 = mul_360 = None
    sub_66: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_189);  sub_65 = unsqueeze_189 = None
    mul_361: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_195);  sub_66 = unsqueeze_195 = None
    mul_362: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_64);  sum_26 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_199: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_198, mul_361);  add_198 = mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(add_199, clone_55, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_55 = primals_93 = None
    getitem_116: "f32[8, 3072, 7, 7]" = convolution_backward_12[0]
    getitem_117: "f32[768, 3072, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_364: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_365: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, convolution_43)
    mul_366: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(mul_365, -0.5);  mul_365 = None
    exp_11: "f32[8, 3072, 7, 7]" = torch.ops.aten.exp.default(mul_366);  mul_366 = None
    mul_367: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_368: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(convolution_43, mul_367);  convolution_43 = mul_367 = None
    add_201: "f32[8, 3072, 7, 7]" = torch.ops.aten.add.Tensor(mul_364, mul_368);  mul_364 = mul_368 = None
    mul_369: "f32[8, 3072, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, add_201);  getitem_116 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_369, add_141, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_369 = add_141 = primals_92 = None
    getitem_119: "f32[8, 768, 7, 7]" = convolution_backward_13[0]
    getitem_120: "f32[3072, 768, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_119, [0, 2, 3])
    sub_67: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_136, unsqueeze_198);  add_136 = unsqueeze_198 = None
    mul_370: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_119, sub_67)
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 2, 3]);  mul_370 = None
    mul_371: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    unsqueeze_199: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_200: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_372: "f32[768]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    mul_373: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_374: "f32[768]" = torch.ops.aten.mul.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
    unsqueeze_202: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_203: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_375: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_90);  primals_90 = None
    unsqueeze_205: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
    unsqueeze_206: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_376: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_204);  sub_67 = unsqueeze_204 = None
    sub_69: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_119, mul_376);  getitem_119 = mul_376 = None
    sub_70: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_201);  sub_69 = unsqueeze_201 = None
    mul_377: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_207);  sub_70 = unsqueeze_207 = None
    mul_378: "f32[768]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_61);  sum_28 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_202: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_199, mul_377);  add_199 = mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(add_202, view_39, primals_89, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_39 = primals_89 = None
    getitem_122: "f32[8, 768, 7, 7]" = convolution_backward_14[0]
    getitem_123: "f32[768, 768, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_94: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(getitem_122, [8, 6, 128, 49]);  getitem_122 = None
    permute_50: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_94, [0, 1, 3, 2]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_95: "f32[48, 49, 128]" = torch.ops.aten.reshape.default(permute_50, [48, 49, 128]);  permute_50 = None
    bmm_28: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(permute_51, view_95);  permute_51 = None
    bmm_29: "f32[48, 49, 49]" = torch.ops.aten.bmm.default(view_95, permute_52);  view_95 = permute_52 = None
    view_96: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_28, [8, 6, 49, 128]);  bmm_28 = None
    view_97: "f32[8, 6, 49, 49]" = torch.ops.aten.reshape.default(bmm_29, [8, 6, 49, 49]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_379: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(view_97, alias_12);  view_97 = None
    sum_29: "f32[8, 6, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [-1], True)
    mul_380: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(alias_12, sum_29);  alias_12 = sum_29 = None
    sub_71: "f32[8, 6, 49, 49]" = torch.ops.aten.sub.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_381: "f32[8, 6, 49, 49]" = torch.ops.aten.mul.Tensor(sub_71, 0.08838834764831845);  sub_71 = None
    view_98: "f32[48, 49, 49]" = torch.ops.aten.reshape.default(mul_381, [48, 49, 49]);  mul_381 = None
    bmm_30: "f32[48, 128, 49]" = torch.ops.aten.bmm.default(permute_53, view_98);  permute_53 = None
    bmm_31: "f32[48, 49, 128]" = torch.ops.aten.bmm.default(view_98, permute_54);  view_98 = permute_54 = None
    view_99: "f32[8, 6, 128, 49]" = torch.ops.aten.reshape.default(bmm_30, [8, 6, 128, 49]);  bmm_30 = None
    view_100: "f32[8, 6, 49, 128]" = torch.ops.aten.reshape.default(bmm_31, [8, 6, 49, 128]);  bmm_31 = None
    permute_55: "f32[8, 6, 49, 128]" = torch.ops.aten.permute.default(view_99, [0, 1, 3, 2]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_3: "f32[24, 6, 49, 128]" = torch.ops.aten.cat.default([view_100, permute_55, view_96]);  view_100 = permute_55 = view_96 = None
    view_101: "f32[3, 8, 6, 49, 128]" = torch.ops.aten.reshape.default(cat_3, [3, 8, 6, 49, 128]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_56: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.permute.default(view_101, [1, 0, 2, 4, 3]);  view_101 = None
    clone_85: "f32[8, 3, 6, 128, 49]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_102: "f32[8, 2304, 7, 7]" = torch.ops.aten.reshape.default(clone_85, [8, 2304, 7, 7]);  clone_85 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(view_102, add_135, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_102 = add_135 = primals_88 = None
    getitem_125: "f32[8, 768, 7, 7]" = convolution_backward_15[0]
    getitem_126: "f32[2304, 768, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(getitem_125, [0, 2, 3])
    sub_72: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(clone_48, unsqueeze_210);  clone_48 = unsqueeze_210 = None
    mul_382: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_125, sub_72)
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[768]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_211: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_212: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_384: "f32[768]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_385: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_386: "f32[768]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_214: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_215: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_387: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_86);  primals_86 = None
    unsqueeze_217: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_218: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_388: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_216);  sub_72 = unsqueeze_216 = None
    sub_74: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_125, mul_388);  getitem_125 = mul_388 = None
    sub_75: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_213);  sub_74 = unsqueeze_213 = None
    mul_389: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_219);  sub_75 = unsqueeze_219 = None
    mul_390: "f32[768]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_58);  sum_31 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_203: "f32[8, 768, 7, 7]" = torch.ops.aten.add.Tensor(add_202, mul_389);  add_202 = mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:421, code: x = self.pos_drop(x + self.pos_embed3)
    sum_32: "f32[1, 768, 7, 7]" = torch.ops.aten.sum.dim_IntList(add_203, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 2, 3])
    sub_76: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_222);  convolution_40 = unsqueeze_222 = None
    mul_391: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(add_203, sub_76)
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_392: "f32[768]" = torch.ops.aten.mul.Tensor(sum_33, 0.002551020408163265)
    unsqueeze_223: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_224: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_393: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    mul_394: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_395: "f32[768]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_226: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_227: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_396: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_84);  primals_84 = None
    unsqueeze_229: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_230: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_397: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_228);  sub_76 = unsqueeze_228 = None
    sub_78: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(add_203, mul_397);  add_203 = mul_397 = None
    sub_79: "f32[8, 768, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_225);  sub_78 = unsqueeze_225 = None
    mul_398: "f32[8, 768, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_231);  sub_79 = unsqueeze_231 = None
    mul_399: "f32[768]" = torch.ops.aten.mul.Tensor(sum_34, squeeze_55);  sum_34 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_398, add_124, primals_82, [768], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_398 = add_124 = primals_82 = None
    getitem_128: "f32[8, 384, 14, 14]" = convolution_backward_16[0]
    getitem_129: "f32[768, 384, 2, 2]" = convolution_backward_16[1]
    getitem_130: "f32[768]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(getitem_128, clone_46, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_46 = primals_81 = None
    getitem_131: "f32[8, 1536, 14, 14]" = convolution_backward_17[0]
    getitem_132: "f32[384, 1536, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_401: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
    mul_402: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, convolution_38)
    mul_403: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_402, -0.5);  mul_402 = None
    exp_12: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_403);  mul_403 = None
    mul_404: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_405: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_38, mul_404);  convolution_38 = mul_404 = None
    add_205: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
    mul_406: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_131, add_205);  getitem_131 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_406, add_122, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = add_122 = primals_80 = None
    getitem_134: "f32[8, 384, 14, 14]" = convolution_backward_18[0]
    getitem_135: "f32[1536, 384, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_134, [0, 2, 3])
    sub_80: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_117, unsqueeze_234);  add_117 = unsqueeze_234 = None
    mul_407: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_134, sub_80)
    sum_36: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 2, 3]);  mul_407 = None
    mul_408: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    unsqueeze_235: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_236: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_409: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    mul_410: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_411: "f32[384]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
    unsqueeze_238: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
    unsqueeze_239: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_412: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_78);  primals_78 = None
    unsqueeze_241: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_242: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_413: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_240);  sub_80 = unsqueeze_240 = None
    sub_82: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_134, mul_413);  getitem_134 = mul_413 = None
    sub_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_237);  sub_82 = unsqueeze_237 = None
    mul_414: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_243);  sub_83 = unsqueeze_243 = None
    mul_415: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, squeeze_52);  sum_36 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_206: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_128, mul_414);  getitem_128 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_206, view_31, primals_77, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_31 = primals_77 = None
    getitem_137: "f32[8, 384, 14, 14]" = convolution_backward_19[0]
    getitem_138: "f32[384, 384, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_103: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_137, [8, 6, 64, 196]);  getitem_137 = None
    permute_57: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_104: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(permute_57, [48, 196, 64]);  permute_57 = None
    bmm_32: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_58, view_104);  permute_58 = None
    bmm_33: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_104, permute_59);  view_104 = permute_59 = None
    view_105: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_32, [8, 6, 196, 64]);  bmm_32 = None
    view_106: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_33, [8, 6, 196, 196]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_416: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_106, alias_13);  view_106 = None
    sum_37: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [-1], True)
    mul_417: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_13, sum_37);  alias_13 = sum_37 = None
    sub_84: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_418: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_84, 0.125);  sub_84 = None
    view_107: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(mul_418, [48, 196, 196]);  mul_418 = None
    bmm_34: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_60, view_107);  permute_60 = None
    bmm_35: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_107, permute_61);  view_107 = permute_61 = None
    view_108: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(bmm_34, [8, 6, 64, 196]);  bmm_34 = None
    view_109: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 6, 196, 64]);  bmm_35 = None
    permute_62: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_108, [0, 1, 3, 2]);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_4: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_109, permute_62, view_105]);  view_109 = permute_62 = view_105 = None
    view_110: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.reshape.default(cat_4, [3, 8, 6, 196, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_63: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_110, [1, 0, 2, 4, 3]);  view_110 = None
    clone_86: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_111: "f32[8, 1152, 14, 14]" = torch.ops.aten.reshape.default(clone_86, [8, 1152, 14, 14]);  clone_86 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(view_111, add_116, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_111 = add_116 = primals_76 = None
    getitem_140: "f32[8, 384, 14, 14]" = convolution_backward_20[0]
    getitem_141: "f32[1152, 384, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_38: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_140, [0, 2, 3])
    sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_111, unsqueeze_246);  add_111 = unsqueeze_246 = None
    mul_419: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_140, sub_85)
    sum_39: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
    mul_420: "f32[384]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_247: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
    unsqueeze_248: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_421: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_422: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_423: "f32[384]" = torch.ops.aten.mul.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
    unsqueeze_250: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_251: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_424: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_74);  primals_74 = None
    unsqueeze_253: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
    unsqueeze_254: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_425: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_252);  sub_85 = unsqueeze_252 = None
    sub_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_140, mul_425);  getitem_140 = mul_425 = None
    sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_249);  sub_87 = unsqueeze_249 = None
    mul_426: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_255);  sub_88 = unsqueeze_255 = None
    mul_427: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_49);  sum_39 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_207: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_206, mul_426);  add_206 = mul_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(add_207, clone_38, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_38 = primals_73 = None
    getitem_143: "f32[8, 1536, 14, 14]" = convolution_backward_21[0]
    getitem_144: "f32[384, 1536, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_429: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
    mul_430: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, convolution_34)
    mul_431: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_430, -0.5);  mul_430 = None
    exp_13: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_431);  mul_431 = None
    mul_432: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_433: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_34, mul_432);  convolution_34 = mul_432 = None
    add_209: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_429, mul_433);  mul_429 = mul_433 = None
    mul_434: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_143, add_209);  getitem_143 = add_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_434, add_109, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = add_109 = primals_72 = None
    getitem_146: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_147: "f32[1536, 384, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_40: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_146, [0, 2, 3])
    sub_89: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_104, unsqueeze_258);  add_104 = unsqueeze_258 = None
    mul_435: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_146, sub_89)
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
    mul_436: "f32[384]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_259: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_260: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_437: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_438: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_439: "f32[384]" = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_262: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
    unsqueeze_263: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_440: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_70);  primals_70 = None
    unsqueeze_265: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_266: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_441: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_264);  sub_89 = unsqueeze_264 = None
    sub_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_146, mul_441);  getitem_146 = mul_441 = None
    sub_92: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_261);  sub_91 = unsqueeze_261 = None
    mul_442: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_267);  sub_92 = unsqueeze_267 = None
    mul_443: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_46);  sum_41 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_210: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_207, mul_442);  add_207 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(add_210, view_23, primals_69, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_23 = primals_69 = None
    getitem_149: "f32[8, 384, 14, 14]" = convolution_backward_23[0]
    getitem_150: "f32[384, 384, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_112: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_149, [8, 6, 64, 196]);  getitem_149 = None
    permute_64: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 2]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_113: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(permute_64, [48, 196, 64]);  permute_64 = None
    bmm_36: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_65, view_113);  permute_65 = None
    bmm_37: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_113, permute_66);  view_113 = permute_66 = None
    view_114: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_36, [8, 6, 196, 64]);  bmm_36 = None
    view_115: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_37, [8, 6, 196, 196]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_444: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_115, alias_14);  view_115 = None
    sum_42: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [-1], True)
    mul_445: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_14, sum_42);  alias_14 = sum_42 = None
    sub_93: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_444, mul_445);  mul_444 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_446: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_93, 0.125);  sub_93 = None
    view_116: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(mul_446, [48, 196, 196]);  mul_446 = None
    bmm_38: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_67, view_116);  permute_67 = None
    bmm_39: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_116, permute_68);  view_116 = permute_68 = None
    view_117: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(bmm_38, [8, 6, 64, 196]);  bmm_38 = None
    view_118: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 6, 196, 64]);  bmm_39 = None
    permute_69: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_5: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_118, permute_69, view_114]);  view_118 = permute_69 = view_114 = None
    view_119: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.reshape.default(cat_5, [3, 8, 6, 196, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_70: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_119, [1, 0, 2, 4, 3]);  view_119 = None
    clone_87: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    view_120: "f32[8, 1152, 14, 14]" = torch.ops.aten.reshape.default(clone_87, [8, 1152, 14, 14]);  clone_87 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(view_120, add_103, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_120 = add_103 = primals_68 = None
    getitem_152: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_153: "f32[1152, 384, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_152, [0, 2, 3])
    sub_94: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_98, unsqueeze_270);  add_98 = unsqueeze_270 = None
    mul_447: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_152, sub_94)
    sum_44: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
    mul_448: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    unsqueeze_271: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_272: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_449: "f32[384]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_451: "f32[384]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    unsqueeze_274: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
    unsqueeze_275: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_452: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_66);  primals_66 = None
    unsqueeze_277: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_278: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_453: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_276);  sub_94 = unsqueeze_276 = None
    sub_96: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_152, mul_453);  getitem_152 = mul_453 = None
    sub_97: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_96, unsqueeze_273);  sub_96 = unsqueeze_273 = None
    mul_454: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_279);  sub_97 = unsqueeze_279 = None
    mul_455: "f32[384]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_43);  sum_44 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_211: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_210, mul_454);  add_210 = mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(add_211, clone_30, primals_65, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_30 = primals_65 = None
    getitem_155: "f32[8, 1536, 14, 14]" = convolution_backward_25[0]
    getitem_156: "f32[384, 1536, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_457: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_458: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, convolution_30)
    mul_459: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_458, -0.5);  mul_458 = None
    exp_14: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_459);  mul_459 = None
    mul_460: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_461: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_30, mul_460);  convolution_30 = mul_460 = None
    add_213: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_457, mul_461);  mul_457 = mul_461 = None
    mul_462: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_155, add_213);  getitem_155 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_462, add_96, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_96 = primals_64 = None
    getitem_158: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_159: "f32[1536, 384, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_158, [0, 2, 3])
    sub_98: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_91, unsqueeze_282);  add_91 = unsqueeze_282 = None
    mul_463: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_158, sub_98)
    sum_46: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    unsqueeze_283: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_284: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_465: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    mul_466: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_467: "f32[384]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_286: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_287: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_468: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_62);  primals_62 = None
    unsqueeze_289: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_290: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_469: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_288);  sub_98 = unsqueeze_288 = None
    sub_100: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_158, mul_469);  getitem_158 = mul_469 = None
    sub_101: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_285);  sub_100 = unsqueeze_285 = None
    mul_470: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_291);  sub_101 = unsqueeze_291 = None
    mul_471: "f32[384]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_40);  sum_46 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_214: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_211, mul_470);  add_211 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(add_214, view_15, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_15 = primals_61 = None
    getitem_161: "f32[8, 384, 14, 14]" = convolution_backward_27[0]
    getitem_162: "f32[384, 384, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_121: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_161, [8, 6, 64, 196]);  getitem_161 = None
    permute_71: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_121, [0, 1, 3, 2]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_122: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(permute_71, [48, 196, 64]);  permute_71 = None
    bmm_40: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_72, view_122);  permute_72 = None
    bmm_41: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_122, permute_73);  view_122 = permute_73 = None
    view_123: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_40, [8, 6, 196, 64]);  bmm_40 = None
    view_124: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_41, [8, 6, 196, 196]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_472: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_124, alias_15);  view_124 = None
    sum_47: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
    mul_473: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_15, sum_47);  alias_15 = sum_47 = None
    sub_102: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_472, mul_473);  mul_472 = mul_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_474: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_102, 0.125);  sub_102 = None
    view_125: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(mul_474, [48, 196, 196]);  mul_474 = None
    bmm_42: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_74, view_125);  permute_74 = None
    bmm_43: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_125, permute_75);  view_125 = permute_75 = None
    view_126: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(bmm_42, [8, 6, 64, 196]);  bmm_42 = None
    view_127: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 6, 196, 64]);  bmm_43 = None
    permute_76: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_126, [0, 1, 3, 2]);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_6: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_127, permute_76, view_123]);  view_127 = permute_76 = view_123 = None
    view_128: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.reshape.default(cat_6, [3, 8, 6, 196, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_77: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_128, [1, 0, 2, 4, 3]);  view_128 = None
    clone_88: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_129: "f32[8, 1152, 14, 14]" = torch.ops.aten.reshape.default(clone_88, [8, 1152, 14, 14]);  clone_88 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(view_129, add_90, primals_60, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_129 = add_90 = primals_60 = None
    getitem_164: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_165: "f32[1152, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_164, [0, 2, 3])
    sub_103: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_85, unsqueeze_294);  add_85 = unsqueeze_294 = None
    mul_475: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_164, sub_103)
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 2, 3]);  mul_475 = None
    mul_476: "f32[384]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_295: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_296: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_477: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_478: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_479: "f32[384]" = torch.ops.aten.mul.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    unsqueeze_298: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
    unsqueeze_299: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_480: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_58);  primals_58 = None
    unsqueeze_301: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_302: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_481: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_300);  sub_103 = unsqueeze_300 = None
    sub_105: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_164, mul_481);  getitem_164 = mul_481 = None
    sub_106: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_297);  sub_105 = unsqueeze_297 = None
    mul_482: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_303);  sub_106 = unsqueeze_303 = None
    mul_483: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_37);  sum_49 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_215: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_214, mul_482);  add_214 = mul_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(add_215, clone_22, primals_57, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  clone_22 = primals_57 = None
    getitem_167: "f32[8, 1536, 14, 14]" = convolution_backward_29[0]
    getitem_168: "f32[384, 1536, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_485: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_486: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, convolution_26)
    mul_487: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
    exp_15: "f32[8, 1536, 14, 14]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
    mul_488: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_489: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(convolution_26, mul_488);  convolution_26 = mul_488 = None
    add_217: "f32[8, 1536, 14, 14]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
    mul_490: "f32[8, 1536, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, add_217);  getitem_167 = add_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_490, add_83, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_490 = add_83 = primals_56 = None
    getitem_170: "f32[8, 384, 14, 14]" = convolution_backward_30[0]
    getitem_171: "f32[1536, 384, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_50: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_170, [0, 2, 3])
    sub_107: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_78, unsqueeze_306);  add_78 = unsqueeze_306 = None
    mul_491: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_170, sub_107)
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[384]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_307: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_308: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_493: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_310: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_311: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_496: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_54);  primals_54 = None
    unsqueeze_313: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_314: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_497: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_312);  sub_107 = unsqueeze_312 = None
    sub_109: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_170, mul_497);  getitem_170 = mul_497 = None
    sub_110: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_309);  sub_109 = unsqueeze_309 = None
    mul_498: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_315);  sub_110 = unsqueeze_315 = None
    mul_499: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_34);  sum_51 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_218: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_215, mul_498);  add_215 = mul_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:107, code: x = self.proj(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_218, view_7, primals_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_7 = primals_53 = None
    getitem_173: "f32[8, 384, 14, 14]" = convolution_backward_31[0]
    getitem_174: "f32[384, 384, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:106, code: x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
    view_130: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(getitem_173, [8, 6, 64, 196]);  getitem_173 = None
    permute_78: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_130, [0, 1, 3, 2]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:104, code: x = attn @ v
    view_131: "f32[48, 196, 64]" = torch.ops.aten.reshape.default(permute_78, [48, 196, 64]);  permute_78 = None
    bmm_44: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(permute_79, view_131);  permute_79 = None
    bmm_45: "f32[48, 196, 196]" = torch.ops.aten.bmm.default(view_131, permute_80);  view_131 = permute_80 = None
    view_132: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_44, [8, 6, 196, 64]);  bmm_44 = None
    view_133: "f32[8, 6, 196, 196]" = torch.ops.aten.reshape.default(bmm_45, [8, 6, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:102, code: attn = attn.softmax(dim=-1)
    mul_500: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(view_133, alias_16);  view_133 = None
    sum_52: "f32[8, 6, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_500, [-1], True)
    mul_501: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(alias_16, sum_52);  alias_16 = sum_52 = None
    sub_111: "f32[8, 6, 196, 196]" = torch.ops.aten.sub.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:101, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_502: "f32[8, 6, 196, 196]" = torch.ops.aten.mul.Tensor(sub_111, 0.125);  sub_111 = None
    view_134: "f32[48, 196, 196]" = torch.ops.aten.reshape.default(mul_502, [48, 196, 196]);  mul_502 = None
    bmm_46: "f32[48, 64, 196]" = torch.ops.aten.bmm.default(permute_81, view_134);  permute_81 = None
    bmm_47: "f32[48, 196, 64]" = torch.ops.aten.bmm.default(view_134, permute_82);  view_134 = permute_82 = None
    view_135: "f32[8, 6, 64, 196]" = torch.ops.aten.reshape.default(bmm_46, [8, 6, 64, 196]);  bmm_46 = None
    view_136: "f32[8, 6, 196, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 6, 196, 64]);  bmm_47 = None
    permute_83: "f32[8, 6, 196, 64]" = torch.ops.aten.permute.default(view_135, [0, 1, 3, 2]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:93, code: q, k, v = x.unbind(0)
    cat_7: "f32[24, 6, 196, 64]" = torch.ops.aten.cat.default([view_136, permute_83, view_132]);  view_136 = permute_83 = view_132 = None
    view_137: "f32[3, 8, 6, 196, 64]" = torch.ops.aten.reshape.default(cat_7, [3, 8, 6, 196, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:92, code: x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
    permute_84: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.permute.default(view_137, [1, 0, 2, 4, 3]);  view_137 = None
    clone_89: "f32[8, 3, 6, 64, 196]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_138: "f32[8, 1152, 14, 14]" = torch.ops.aten.reshape.default(clone_89, [8, 1152, 14, 14]);  clone_89 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(view_138, add_77, primals_52, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_138 = add_77 = primals_52 = None
    getitem_176: "f32[8, 384, 14, 14]" = convolution_backward_32[0]
    getitem_177: "f32[1152, 384, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    sum_53: "f32[384]" = torch.ops.aten.sum.dim_IntList(getitem_176, [0, 2, 3])
    sub_112: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(clone_15, unsqueeze_318);  clone_15 = unsqueeze_318 = None
    mul_503: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, sub_112)
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    unsqueeze_319: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_320: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_505: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    mul_506: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_322: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_323: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_508: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_50);  primals_50 = None
    unsqueeze_325: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_326: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_509: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_324);  sub_112 = unsqueeze_324 = None
    sub_114: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_176, mul_509);  getitem_176 = mul_509 = None
    sub_115: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_321);  sub_114 = unsqueeze_321 = None
    mul_510: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_327);  sub_115 = unsqueeze_327 = None
    mul_511: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, squeeze_31);  sum_54 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:156, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    add_219: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_218, mul_510);  add_218 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:411, code: x = self.pos_drop(x + self.pos_embed2)
    sum_55: "f32[1, 384, 14, 14]" = torch.ops.aten.sum.dim_IntList(add_219, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 2, 3])
    sub_116: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_330);  convolution_23 = unsqueeze_330 = None
    mul_512: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(add_219, sub_116)
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
    mul_513: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_331: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_332: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_514: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_515: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_516: "f32[384]" = torch.ops.aten.mul.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    unsqueeze_334: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_335: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_517: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_48);  primals_48 = None
    unsqueeze_337: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_338: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_518: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_336);  sub_116 = unsqueeze_336 = None
    sub_118: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(add_219, mul_518);  add_219 = mul_518 = None
    sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_333);  sub_118 = unsqueeze_333 = None
    mul_519: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_339);  sub_119 = unsqueeze_339 = None
    mul_520: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_28);  sum_57 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_519, add_66, primals_46, [384], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_519 = add_66 = primals_46 = None
    getitem_179: "f32[8, 192, 28, 28]" = convolution_backward_33[0]
    getitem_180: "f32[384, 192, 2, 2]" = convolution_backward_33[1]
    getitem_181: "f32[384]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(getitem_179, mul_104, primals_45, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_104 = primals_45 = None
    getitem_182: "f32[8, 384, 28, 28]" = convolution_backward_34[0]
    getitem_183: "f32[192, 384, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_522: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_65, 0.5);  add_65 = None
    mul_523: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, convolution_21)
    mul_524: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_523, -0.5);  mul_523 = None
    exp_16: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_524);  mul_524 = None
    mul_525: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_526: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_21, mul_525);  convolution_21 = mul_525 = None
    add_221: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_522, mul_526);  mul_522 = mul_526 = None
    mul_527: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_182, add_221);  getitem_182 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_527, clone_13, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_527 = clone_13 = primals_44 = None
    getitem_185: "f32[8, 384, 28, 28]" = convolution_backward_35[0]
    getitem_186: "f32[384, 48, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_529: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_530: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, convolution_20)
    mul_531: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_530, -0.5);  mul_530 = None
    exp_17: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_531);  mul_531 = None
    mul_532: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_533: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_20, mul_532);  convolution_20 = mul_532 = None
    add_223: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_529, mul_533);  mul_529 = mul_533 = None
    mul_534: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_185, add_223);  getitem_185 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_534, add_63, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_534 = add_63 = primals_43 = None
    getitem_188: "f32[8, 192, 28, 28]" = convolution_backward_36[0]
    getitem_189: "f32[384, 192, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_58: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_188, [0, 2, 3])
    sub_120: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_58, unsqueeze_342);  add_58 = unsqueeze_342 = None
    mul_535: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_188, sub_120)
    sum_59: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[192]" = torch.ops.aten.mul.Tensor(sum_58, 0.00015943877551020407)
    unsqueeze_343: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_344: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_537: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, 0.00015943877551020407)
    mul_538: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_539: "f32[192]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_346: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_347: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_540: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_41);  primals_41 = None
    unsqueeze_349: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_350: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_541: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_348);  sub_120 = unsqueeze_348 = None
    sub_122: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_188, mul_541);  getitem_188 = mul_541 = None
    sub_123: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_345);  sub_122 = unsqueeze_345 = None
    mul_542: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_351);  sub_123 = unsqueeze_351 = None
    mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_25);  sum_59 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_224: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_179, mul_542);  getitem_179 = mul_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(add_224, mul_91, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_91 = primals_40 = None
    getitem_191: "f32[8, 384, 28, 28]" = convolution_backward_37[0]
    getitem_192: "f32[192, 384, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_545: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
    mul_546: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, convolution_18)
    mul_547: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_546, -0.5);  mul_546 = None
    exp_18: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_547);  mul_547 = None
    mul_548: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_549: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_18, mul_548);  convolution_18 = mul_548 = None
    add_226: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_545, mul_549);  mul_545 = mul_549 = None
    mul_550: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_191, add_226);  getitem_191 = add_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_550, clone_11, primals_39, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_550 = clone_11 = primals_39 = None
    getitem_194: "f32[8, 384, 28, 28]" = convolution_backward_38[0]
    getitem_195: "f32[384, 48, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_552: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_553: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, convolution_17)
    mul_554: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_553, -0.5);  mul_553 = None
    exp_19: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_554);  mul_554 = None
    mul_555: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_556: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_17, mul_555);  convolution_17 = mul_555 = None
    add_228: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_552, mul_556);  mul_552 = mul_556 = None
    mul_557: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_194, add_228);  getitem_194 = add_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_557, add_55, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_557 = add_55 = primals_38 = None
    getitem_197: "f32[8, 192, 28, 28]" = convolution_backward_39[0]
    getitem_198: "f32[384, 192, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_60: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_197, [0, 2, 3])
    sub_124: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_50, unsqueeze_354);  add_50 = unsqueeze_354 = None
    mul_558: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_197, sub_124)
    sum_61: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3]);  mul_558 = None
    mul_559: "f32[192]" = torch.ops.aten.mul.Tensor(sum_60, 0.00015943877551020407)
    unsqueeze_355: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_356: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_560: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, 0.00015943877551020407)
    mul_561: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_562: "f32[192]" = torch.ops.aten.mul.Tensor(mul_560, mul_561);  mul_560 = mul_561 = None
    unsqueeze_358: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_359: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_563: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_36);  primals_36 = None
    unsqueeze_361: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_362: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_564: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_360);  sub_124 = unsqueeze_360 = None
    sub_126: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_197, mul_564);  getitem_197 = mul_564 = None
    sub_127: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_357);  sub_126 = unsqueeze_357 = None
    mul_565: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_363);  sub_127 = unsqueeze_363 = None
    mul_566: "f32[192]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_22);  sum_61 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_229: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_224, mul_565);  add_224 = mul_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(add_229, mul_78, primals_35, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_78 = primals_35 = None
    getitem_200: "f32[8, 384, 28, 28]" = convolution_backward_40[0]
    getitem_201: "f32[192, 384, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_568: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_569: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, convolution_15)
    mul_570: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_569, -0.5);  mul_569 = None
    exp_20: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_570);  mul_570 = None
    mul_571: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_572: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_15, mul_571);  convolution_15 = mul_571 = None
    add_231: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_568, mul_572);  mul_568 = mul_572 = None
    mul_573: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_200, add_231);  getitem_200 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_573, clone_9, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_573 = clone_9 = primals_34 = None
    getitem_203: "f32[8, 384, 28, 28]" = convolution_backward_41[0]
    getitem_204: "f32[384, 48, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_575: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_576: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, convolution_14)
    mul_577: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_576, -0.5);  mul_576 = None
    exp_21: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_577);  mul_577 = None
    mul_578: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_579: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_14, mul_578);  convolution_14 = mul_578 = None
    add_233: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_575, mul_579);  mul_575 = mul_579 = None
    mul_580: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_203, add_233);  getitem_203 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_580, add_47, primals_33, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_580 = add_47 = primals_33 = None
    getitem_206: "f32[8, 192, 28, 28]" = convolution_backward_42[0]
    getitem_207: "f32[384, 192, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_62: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_206, [0, 2, 3])
    sub_128: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_42, unsqueeze_366);  add_42 = unsqueeze_366 = None
    mul_581: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_206, sub_128)
    sum_63: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
    mul_582: "f32[192]" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
    unsqueeze_367: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_368: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_583: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
    mul_584: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_585: "f32[192]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    unsqueeze_370: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_371: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_586: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_31);  primals_31 = None
    unsqueeze_373: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_374: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_587: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_372);  sub_128 = unsqueeze_372 = None
    sub_130: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_206, mul_587);  getitem_206 = mul_587 = None
    sub_131: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_369);  sub_130 = unsqueeze_369 = None
    mul_588: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_375);  sub_131 = unsqueeze_375 = None
    mul_589: "f32[192]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_19);  sum_63 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_234: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_229, mul_588);  add_229 = mul_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(add_234, mul_65, primals_30, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_65 = primals_30 = None
    getitem_209: "f32[8, 384, 28, 28]" = convolution_backward_43[0]
    getitem_210: "f32[192, 384, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_591: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_592: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
    mul_593: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_592, -0.5);  mul_592 = None
    exp_22: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_593);  mul_593 = None
    mul_594: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_595: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_12, mul_594);  convolution_12 = mul_594 = None
    add_236: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_591, mul_595);  mul_591 = mul_595 = None
    mul_596: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_209, add_236);  getitem_209 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_596, clone_7, primals_29, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_596 = clone_7 = primals_29 = None
    getitem_212: "f32[8, 384, 28, 28]" = convolution_backward_44[0]
    getitem_213: "f32[384, 48, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_598: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_599: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, convolution_11)
    mul_600: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_599, -0.5);  mul_599 = None
    exp_23: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_600);  mul_600 = None
    mul_601: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_602: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_11, mul_601);  convolution_11 = mul_601 = None
    add_238: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_598, mul_602);  mul_598 = mul_602 = None
    mul_603: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_212, add_238);  getitem_212 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_603, add_39, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = add_39 = primals_28 = None
    getitem_215: "f32[8, 192, 28, 28]" = convolution_backward_45[0]
    getitem_216: "f32[384, 192, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_64: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_215, [0, 2, 3])
    sub_132: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_34, unsqueeze_378);  add_34 = unsqueeze_378 = None
    mul_604: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_215, sub_132)
    sum_65: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 2, 3]);  mul_604 = None
    mul_605: "f32[192]" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
    unsqueeze_379: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_380: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_606: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
    mul_607: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_608: "f32[192]" = torch.ops.aten.mul.Tensor(mul_606, mul_607);  mul_606 = mul_607 = None
    unsqueeze_382: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_383: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_609: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_26);  primals_26 = None
    unsqueeze_385: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_386: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_610: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_384);  sub_132 = unsqueeze_384 = None
    sub_134: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_215, mul_610);  getitem_215 = mul_610 = None
    sub_135: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_381);  sub_134 = unsqueeze_381 = None
    mul_611: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_387);  sub_135 = unsqueeze_387 = None
    mul_612: "f32[192]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_16);  sum_65 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_239: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_234, mul_611);  add_234 = mul_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(add_239, mul_52, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_52 = primals_25 = None
    getitem_218: "f32[8, 384, 28, 28]" = convolution_backward_46[0]
    getitem_219: "f32[192, 384, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_614: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
    mul_615: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, convolution_9)
    mul_616: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_615, -0.5);  mul_615 = None
    exp_24: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_616);  mul_616 = None
    mul_617: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_618: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_9, mul_617);  convolution_9 = mul_617 = None
    add_241: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_614, mul_618);  mul_614 = mul_618 = None
    mul_619: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_218, add_241);  getitem_218 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_619, clone_5, primals_24, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_619 = clone_5 = primals_24 = None
    getitem_221: "f32[8, 384, 28, 28]" = convolution_backward_47[0]
    getitem_222: "f32[384, 48, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_621: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_622: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, convolution_8)
    mul_623: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_622, -0.5);  mul_622 = None
    exp_25: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_623);  mul_623 = None
    mul_624: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_625: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_8, mul_624);  convolution_8 = mul_624 = None
    add_243: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_621, mul_625);  mul_621 = mul_625 = None
    mul_626: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_221, add_243);  getitem_221 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_626, add_31, primals_23, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = add_31 = primals_23 = None
    getitem_224: "f32[8, 192, 28, 28]" = convolution_backward_48[0]
    getitem_225: "f32[384, 192, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_66: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_224, [0, 2, 3])
    sub_136: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_26, unsqueeze_390);  add_26 = unsqueeze_390 = None
    mul_627: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_224, sub_136)
    sum_67: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
    mul_628: "f32[192]" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
    unsqueeze_391: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_392: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_629: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
    mul_630: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_631: "f32[192]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
    unsqueeze_394: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_395: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_632: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_21);  primals_21 = None
    unsqueeze_397: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_398: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_633: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_396);  sub_136 = unsqueeze_396 = None
    sub_138: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_224, mul_633);  getitem_224 = mul_633 = None
    sub_139: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_393);  sub_138 = unsqueeze_393 = None
    mul_634: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_399);  sub_139 = unsqueeze_399 = None
    mul_635: "f32[192]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_13);  sum_67 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_244: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_239, mul_634);  add_239 = mul_634 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(add_244, mul_39, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_39 = primals_20 = None
    getitem_227: "f32[8, 384, 28, 28]" = convolution_backward_49[0]
    getitem_228: "f32[192, 384, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_637: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
    mul_638: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, convolution_6)
    mul_639: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_638, -0.5);  mul_638 = None
    exp_26: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_639);  mul_639 = None
    mul_640: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_641: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_6, mul_640);  convolution_6 = mul_640 = None
    add_246: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_637, mul_641);  mul_637 = mul_641 = None
    mul_642: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_227, add_246);  getitem_227 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_642, clone_3, primals_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_642 = clone_3 = primals_19 = None
    getitem_230: "f32[8, 384, 28, 28]" = convolution_backward_50[0]
    getitem_231: "f32[384, 48, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_644: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_645: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
    mul_646: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_645, -0.5);  mul_645 = None
    exp_27: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_646);  mul_646 = None
    mul_647: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_648: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_5, mul_647);  convolution_5 = mul_647 = None
    add_248: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_644, mul_648);  mul_644 = mul_648 = None
    mul_649: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_230, add_248);  getitem_230 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_649, add_23, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_649 = add_23 = primals_18 = None
    getitem_233: "f32[8, 192, 28, 28]" = convolution_backward_51[0]
    getitem_234: "f32[384, 192, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_68: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_233, [0, 2, 3])
    sub_140: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_18, unsqueeze_402);  add_18 = unsqueeze_402 = None
    mul_650: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_233, sub_140)
    sum_69: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3]);  mul_650 = None
    mul_651: "f32[192]" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
    unsqueeze_403: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_404: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_652: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
    mul_653: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_654: "f32[192]" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
    unsqueeze_406: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_407: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_655: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_16);  primals_16 = None
    unsqueeze_409: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_410: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_656: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_408);  sub_140 = unsqueeze_408 = None
    sub_142: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_233, mul_656);  getitem_233 = mul_656 = None
    sub_143: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_405);  sub_142 = unsqueeze_405 = None
    mul_657: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_411);  sub_143 = unsqueeze_411 = None
    mul_658: "f32[192]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_10);  sum_69 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_249: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_244, mul_657);  add_244 = mul_657 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:68, code: x = self.conv3(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(add_249, mul_26, primals_15, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_26 = primals_15 = None
    getitem_236: "f32[8, 384, 28, 28]" = convolution_backward_52[0]
    getitem_237: "f32[192, 384, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:67, code: x = self.act2(x)
    mul_660: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
    mul_661: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, convolution_3)
    mul_662: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_661, -0.5);  mul_661 = None
    exp_28: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_662);  mul_662 = None
    mul_663: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_664: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_3, mul_663);  convolution_3 = mul_663 = None
    add_251: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_660, mul_664);  mul_660 = mul_664 = None
    mul_665: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_236, add_251);  getitem_236 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:66, code: x = self.conv2(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_665, clone_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False]);  mul_665 = clone_1 = primals_14 = None
    getitem_239: "f32[8, 384, 28, 28]" = convolution_backward_53[0]
    getitem_240: "f32[384, 48, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:63, code: x = self.act1(x)
    mul_667: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_668: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
    mul_669: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(mul_668, -0.5);  mul_668 = None
    exp_29: "f32[8, 384, 28, 28]" = torch.ops.aten.exp.default(mul_669);  mul_669 = None
    mul_670: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_671: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(convolution_2, mul_670);  convolution_2 = mul_670 = None
    add_253: "f32[8, 384, 28, 28]" = torch.ops.aten.add.Tensor(mul_667, mul_671);  mul_667 = mul_671 = None
    mul_672: "f32[8, 384, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_239, add_253);  getitem_239 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:62, code: x = self.conv1(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_672, add_15, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_672 = add_15 = primals_13 = None
    getitem_242: "f32[8, 192, 28, 28]" = convolution_backward_54[0]
    getitem_243: "f32[384, 192, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    sum_70: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_242, [0, 2, 3])
    sub_144: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(clone, unsqueeze_414);  clone = unsqueeze_414 = None
    mul_673: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_242, sub_144)
    sum_71: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 2, 3]);  mul_673 = None
    mul_674: "f32[192]" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
    unsqueeze_415: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_416: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_675: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
    mul_676: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_677: "f32[192]" = torch.ops.aten.mul.Tensor(mul_675, mul_676);  mul_675 = mul_676 = None
    unsqueeze_418: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_419: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_678: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_11);  primals_11 = None
    unsqueeze_421: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_422: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_679: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_420);  sub_144 = unsqueeze_420 = None
    sub_146: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_242, mul_679);  getitem_242 = mul_679 = None
    sub_147: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_417);  sub_146 = unsqueeze_417 = None
    mul_680: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_423);  sub_147 = unsqueeze_423 = None
    mul_681: "f32[192]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_7);  sum_71 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:157, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    add_254: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_249, mul_680);  add_249 = mul_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:401, code: x = self.pos_drop(x + self.pos_embed1)
    sum_72: "f32[1, 192, 28, 28]" = torch.ops.aten.sum.dim_IntList(add_254, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 2, 3])
    sub_148: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_426);  convolution_1 = unsqueeze_426 = None
    mul_682: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(add_254, sub_148)
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
    mul_683: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
    unsqueeze_427: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_428: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_684: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
    mul_685: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_686: "f32[192]" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
    unsqueeze_430: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_431: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_687: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_9);  primals_9 = None
    unsqueeze_433: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
    unsqueeze_434: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_688: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_432);  sub_148 = unsqueeze_432 = None
    sub_150: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(add_254, mul_688);  add_254 = mul_688 = None
    sub_151: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_429);  sub_150 = unsqueeze_429 = None
    mul_689: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_435);  sub_151 = unsqueeze_435 = None
    mul_690: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, squeeze_4);  sum_74 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_689, relu, primals_7, [192], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  mul_689 = primals_7 = None
    getitem_245: "f32[8, 32, 112, 112]" = convolution_backward_55[0]
    getitem_246: "f32[192, 32, 4, 4]" = convolution_backward_55[1]
    getitem_247: "f32[192]" = convolution_backward_55[2];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/visformer.py:396, code: x = self.stem(x)
    le: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le, full_default, getitem_245);  le = full_default = getitem_245 = None
    sum_75: "f32[32]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_152: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_438);  convolution = unsqueeze_438 = None
    mul_691: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where, sub_152)
    sum_76: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_692: "f32[32]" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
    unsqueeze_439: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_440: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_693: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, 9.964923469387754e-06)
    mul_694: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_695: "f32[32]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_442: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_443: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_696: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_5);  primals_5 = None
    unsqueeze_445: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_446: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_697: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_444);  sub_152 = unsqueeze_444 = None
    sub_154: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where, mul_697);  where = mul_697 = None
    sub_155: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_441);  sub_154 = unsqueeze_441 = None
    mul_698: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_447);  sub_155 = unsqueeze_447 = None
    mul_699: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, squeeze_1);  sum_76 = squeeze_1 = None
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_698, primals_206, primals_4, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_698 = primals_206 = primals_4 = None
    getitem_249: "f32[32, 3, 7, 7]" = convolution_backward_56[1];  convolution_backward_56 = None
    return [sum_72, sum_55, sum_32, getitem_249, mul_699, sum_75, getitem_246, getitem_247, mul_690, sum_73, mul_681, sum_70, getitem_243, getitem_240, getitem_237, mul_658, sum_68, getitem_234, getitem_231, getitem_228, mul_635, sum_66, getitem_225, getitem_222, getitem_219, mul_612, sum_64, getitem_216, getitem_213, getitem_210, mul_589, sum_62, getitem_207, getitem_204, getitem_201, mul_566, sum_60, getitem_198, getitem_195, getitem_192, mul_543, sum_58, getitem_189, getitem_186, getitem_183, getitem_180, getitem_181, mul_520, sum_56, mul_511, sum_53, getitem_177, getitem_174, mul_499, sum_50, getitem_171, getitem_168, mul_483, sum_48, getitem_165, getitem_162, mul_471, sum_45, getitem_159, getitem_156, mul_455, sum_43, getitem_153, getitem_150, mul_443, sum_40, getitem_147, getitem_144, mul_427, sum_38, getitem_141, getitem_138, mul_415, sum_35, getitem_135, getitem_132, getitem_129, getitem_130, mul_399, sum_33, mul_390, sum_30, getitem_126, getitem_123, mul_378, sum_27, getitem_120, getitem_117, mul_362, sum_25, getitem_114, getitem_111, mul_350, sum_22, getitem_108, getitem_105, mul_334, sum_20, getitem_102, getitem_99, mul_322, sum_17, getitem_96, getitem_93, mul_306, sum_15, getitem_90, getitem_87, mul_294, sum_12, getitem_84, getitem_81, mul_278, sum_10, permute_28, view_65, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    