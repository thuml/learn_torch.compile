from __future__ import annotations



def forward(self, primals_4: "f32[768, 3, 16, 16]", primals_6: "f32[768]", primals_12: "f32[768]", primals_18: "f32[768]", primals_24: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_42: "f32[768]", primals_48: "f32[768]", primals_54: "f32[768]", primals_60: "f32[768]", primals_66: "f32[768]", primals_72: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_90: "f32[768]", primals_96: "f32[768]", primals_102: "f32[768]", primals_108: "f32[768]", primals_114: "f32[768]", primals_120: "f32[768]", primals_126: "f32[768]", primals_132: "f32[768]", primals_138: "f32[768]", primals_144: "f32[768]", primals_150: "f32[768]", primals_156: "f32[8, 3, 224, 224]", mul: "f32[8, 198, 768]", view_1: "f32[1584, 768]", getitem_2: "f32[8, 12, 198, 64]", getitem_3: "f32[8, 12, 198, 64]", getitem_4: "f32[8, 12, 198, 64]", getitem_6: "f32[8, 12, 224]", getitem_7: "i64[]", getitem_8: "i64[]", view_5: "f32[1584, 768]", mul_2: "f32[8, 198, 768]", view_7: "f32[1584, 768]", addmm_2: "f32[1584, 3072]", view_9: "f32[1584, 3072]", mul_7: "f32[8, 198, 768]", view_11: "f32[1584, 768]", getitem_13: "f32[8, 12, 198, 64]", getitem_14: "f32[8, 12, 198, 64]", getitem_15: "f32[8, 12, 198, 64]", getitem_17: "f32[8, 12, 224]", getitem_18: "i64[]", getitem_19: "i64[]", view_15: "f32[1584, 768]", mul_9: "f32[8, 198, 768]", view_17: "f32[1584, 768]", addmm_6: "f32[1584, 3072]", view_19: "f32[1584, 3072]", mul_14: "f32[8, 198, 768]", view_21: "f32[1584, 768]", getitem_24: "f32[8, 12, 198, 64]", getitem_25: "f32[8, 12, 198, 64]", getitem_26: "f32[8, 12, 198, 64]", getitem_28: "f32[8, 12, 224]", getitem_29: "i64[]", getitem_30: "i64[]", view_25: "f32[1584, 768]", mul_16: "f32[8, 198, 768]", view_27: "f32[1584, 768]", addmm_10: "f32[1584, 3072]", view_29: "f32[1584, 3072]", mul_21: "f32[8, 198, 768]", view_31: "f32[1584, 768]", getitem_35: "f32[8, 12, 198, 64]", getitem_36: "f32[8, 12, 198, 64]", getitem_37: "f32[8, 12, 198, 64]", getitem_39: "f32[8, 12, 224]", getitem_40: "i64[]", getitem_41: "i64[]", view_35: "f32[1584, 768]", mul_23: "f32[8, 198, 768]", view_37: "f32[1584, 768]", addmm_14: "f32[1584, 3072]", view_39: "f32[1584, 3072]", mul_28: "f32[8, 198, 768]", view_41: "f32[1584, 768]", getitem_46: "f32[8, 12, 198, 64]", getitem_47: "f32[8, 12, 198, 64]", getitem_48: "f32[8, 12, 198, 64]", getitem_50: "f32[8, 12, 224]", getitem_51: "i64[]", getitem_52: "i64[]", view_45: "f32[1584, 768]", mul_30: "f32[8, 198, 768]", view_47: "f32[1584, 768]", addmm_18: "f32[1584, 3072]", view_49: "f32[1584, 3072]", mul_35: "f32[8, 198, 768]", view_51: "f32[1584, 768]", getitem_57: "f32[8, 12, 198, 64]", getitem_58: "f32[8, 12, 198, 64]", getitem_59: "f32[8, 12, 198, 64]", getitem_61: "f32[8, 12, 224]", getitem_62: "i64[]", getitem_63: "i64[]", view_55: "f32[1584, 768]", mul_37: "f32[8, 198, 768]", view_57: "f32[1584, 768]", addmm_22: "f32[1584, 3072]", view_59: "f32[1584, 3072]", mul_42: "f32[8, 198, 768]", view_61: "f32[1584, 768]", getitem_68: "f32[8, 12, 198, 64]", getitem_69: "f32[8, 12, 198, 64]", getitem_70: "f32[8, 12, 198, 64]", getitem_72: "f32[8, 12, 224]", getitem_73: "i64[]", getitem_74: "i64[]", view_65: "f32[1584, 768]", mul_44: "f32[8, 198, 768]", view_67: "f32[1584, 768]", addmm_26: "f32[1584, 3072]", view_69: "f32[1584, 3072]", mul_49: "f32[8, 198, 768]", view_71: "f32[1584, 768]", getitem_79: "f32[8, 12, 198, 64]", getitem_80: "f32[8, 12, 198, 64]", getitem_81: "f32[8, 12, 198, 64]", getitem_83: "f32[8, 12, 224]", getitem_84: "i64[]", getitem_85: "i64[]", view_75: "f32[1584, 768]", mul_51: "f32[8, 198, 768]", view_77: "f32[1584, 768]", addmm_30: "f32[1584, 3072]", view_79: "f32[1584, 3072]", mul_56: "f32[8, 198, 768]", view_81: "f32[1584, 768]", getitem_90: "f32[8, 12, 198, 64]", getitem_91: "f32[8, 12, 198, 64]", getitem_92: "f32[8, 12, 198, 64]", getitem_94: "f32[8, 12, 224]", getitem_95: "i64[]", getitem_96: "i64[]", view_85: "f32[1584, 768]", mul_58: "f32[8, 198, 768]", view_87: "f32[1584, 768]", addmm_34: "f32[1584, 3072]", view_89: "f32[1584, 3072]", mul_63: "f32[8, 198, 768]", view_91: "f32[1584, 768]", getitem_101: "f32[8, 12, 198, 64]", getitem_102: "f32[8, 12, 198, 64]", getitem_103: "f32[8, 12, 198, 64]", getitem_105: "f32[8, 12, 224]", getitem_106: "i64[]", getitem_107: "i64[]", view_95: "f32[1584, 768]", mul_65: "f32[8, 198, 768]", view_97: "f32[1584, 768]", addmm_38: "f32[1584, 3072]", view_99: "f32[1584, 3072]", mul_70: "f32[8, 198, 768]", view_101: "f32[1584, 768]", getitem_112: "f32[8, 12, 198, 64]", getitem_113: "f32[8, 12, 198, 64]", getitem_114: "f32[8, 12, 198, 64]", getitem_116: "f32[8, 12, 224]", getitem_117: "i64[]", getitem_118: "i64[]", view_105: "f32[1584, 768]", mul_72: "f32[8, 198, 768]", view_107: "f32[1584, 768]", addmm_42: "f32[1584, 3072]", view_109: "f32[1584, 3072]", mul_77: "f32[8, 198, 768]", view_111: "f32[1584, 768]", getitem_123: "f32[8, 12, 198, 64]", getitem_124: "f32[8, 12, 198, 64]", getitem_125: "f32[8, 12, 198, 64]", getitem_127: "f32[8, 12, 224]", getitem_128: "i64[]", getitem_129: "i64[]", view_115: "f32[1584, 768]", mul_79: "f32[8, 198, 768]", view_117: "f32[1584, 768]", addmm_46: "f32[1584, 3072]", view_119: "f32[1584, 3072]", mul_84: "f32[8, 198, 768]", select: "f32[8, 768]", select_1: "f32[8, 768]", permute_75: "f32[1000, 768]", permute_79: "f32[1000, 768]", div_2: "f32[8, 198, 1]", permute_83: "f32[768, 3072]", permute_87: "f32[3072, 768]", div_3: "f32[8, 198, 1]", permute_91: "f32[768, 768]", alias_12: "f32[8, 12, 198, 64]", permute_97: "f32[2304, 768]", div_4: "f32[8, 198, 1]", permute_101: "f32[768, 3072]", permute_105: "f32[3072, 768]", div_5: "f32[8, 198, 1]", permute_109: "f32[768, 768]", alias_13: "f32[8, 12, 198, 64]", permute_115: "f32[2304, 768]", div_6: "f32[8, 198, 1]", permute_119: "f32[768, 3072]", permute_123: "f32[3072, 768]", div_7: "f32[8, 198, 1]", permute_127: "f32[768, 768]", alias_14: "f32[8, 12, 198, 64]", permute_133: "f32[2304, 768]", div_8: "f32[8, 198, 1]", permute_137: "f32[768, 3072]", permute_141: "f32[3072, 768]", div_9: "f32[8, 198, 1]", permute_145: "f32[768, 768]", alias_15: "f32[8, 12, 198, 64]", permute_151: "f32[2304, 768]", div_10: "f32[8, 198, 1]", permute_155: "f32[768, 3072]", permute_159: "f32[3072, 768]", div_11: "f32[8, 198, 1]", permute_163: "f32[768, 768]", alias_16: "f32[8, 12, 198, 64]", permute_169: "f32[2304, 768]", div_12: "f32[8, 198, 1]", permute_173: "f32[768, 3072]", permute_177: "f32[3072, 768]", div_13: "f32[8, 198, 1]", permute_181: "f32[768, 768]", alias_17: "f32[8, 12, 198, 64]", permute_187: "f32[2304, 768]", div_14: "f32[8, 198, 1]", permute_191: "f32[768, 3072]", permute_195: "f32[3072, 768]", div_15: "f32[8, 198, 1]", permute_199: "f32[768, 768]", alias_18: "f32[8, 12, 198, 64]", permute_205: "f32[2304, 768]", div_16: "f32[8, 198, 1]", permute_209: "f32[768, 3072]", permute_213: "f32[3072, 768]", div_17: "f32[8, 198, 1]", permute_217: "f32[768, 768]", alias_19: "f32[8, 12, 198, 64]", permute_223: "f32[2304, 768]", div_18: "f32[8, 198, 1]", permute_227: "f32[768, 3072]", permute_231: "f32[3072, 768]", div_19: "f32[8, 198, 1]", permute_235: "f32[768, 768]", alias_20: "f32[8, 12, 198, 64]", permute_241: "f32[2304, 768]", div_20: "f32[8, 198, 1]", permute_245: "f32[768, 3072]", permute_249: "f32[3072, 768]", div_21: "f32[8, 198, 1]", permute_253: "f32[768, 768]", alias_21: "f32[8, 12, 198, 64]", permute_259: "f32[2304, 768]", div_22: "f32[8, 198, 1]", permute_263: "f32[768, 3072]", permute_267: "f32[3072, 768]", div_23: "f32[8, 198, 1]", permute_271: "f32[768, 768]", alias_22: "f32[8, 12, 198, 64]", permute_277: "f32[2304, 768]", div_24: "f32[8, 198, 1]", permute_281: "f32[768, 3072]", permute_285: "f32[3072, 768]", div_25: "f32[8, 198, 1]", permute_289: "f32[768, 768]", alias_23: "f32[8, 12, 198, 64]", permute_295: "f32[2304, 768]", div_26: "f32[8, 198, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_8: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 198, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 198, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_12: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_1: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_28: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 198, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_2: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 198, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_3: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_48: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 198, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_33: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_4: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 198, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_5: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 198, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_6: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 198, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_54: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_7: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_55: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_88: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 198, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_61: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_8: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_62: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_98: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 198, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_9: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 198, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_10: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_76: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 198, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_11: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_83: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:118, code: return (x + x_dist) / 2
    div_1: "f32[8, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:112, code: x_dist = self.head_dist(x_dist)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(div_1, permute_75);  permute_75 = None
    permute_76: "f32[1000, 8]" = torch.ops.aten.permute.default(div_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_76, select_1);  select_1 = None
    permute_77: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_1, [0], True)
    view_121: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_78: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:111, code: x = self.head(x)
    mm_2: "f32[8, 768]" = torch.ops.aten.mm.default(div_1, permute_79);  div_1 = permute_79 = None
    mm_3: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_76, select);  permute_76 = select = None
    permute_81: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    permute_82: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    full_default: "f32[8, 198, 768]" = torch.ops.aten.full.default([8, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 1);  mm = None
    select_scatter_1: "f32[8, 198, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_2, 1, 0);  full_default = mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    add_88: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(select_scatter, select_scatter_1);  select_scatter = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    mul_87: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(add_88, primals_150);  primals_150 = None
    mul_88: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_87, 768)
    sum_3: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
    mul_89: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_87, mul_84);  mul_87 = None
    sum_4: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
    mul_90: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_4);  sum_4 = None
    sub_26: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_88, sum_3);  mul_88 = sum_3 = None
    sub_27: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
    mul_91: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_27);  div_2 = sub_27 = None
    mul_92: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_84);  mul_84 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
    sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_123: "f32[1584, 768]" = torch.ops.aten.reshape.default(mul_91, [1584, 768])
    mm_4: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_123, permute_83);  permute_83 = None
    permute_84: "f32[768, 1584]" = torch.ops.aten.permute.default(view_123, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_84, view_119);  permute_84 = view_119 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
    view_124: "f32[768]" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    view_125: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_4, [8, 198, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_94: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
    mul_95: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_96: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
    exp: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
    mul_97: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_98: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_97);  view_118 = mul_97 = None
    add_90: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
    mul_99: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_125, add_90);  view_125 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_99, [1584, 3072]);  mul_99 = None
    mm_6: "f32[1584, 768]" = torch.ops.aten.mm.default(view_126, permute_87);  permute_87 = None
    permute_88: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_88, view_117);  permute_88 = view_117 = None
    permute_89: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_8: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[3072]" = torch.ops.aten.reshape.default(sum_8, [3072]);  sum_8 = None
    permute_90: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    view_128: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 198, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_101: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_128, primals_144);  primals_144 = None
    mul_102: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
    sum_9: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
    mul_103: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_79);  mul_101 = None
    sum_10: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
    mul_104: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_10);  sum_10 = None
    sub_29: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_9);  mul_102 = sum_9 = None
    sub_30: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
    mul_105: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_30);  div_3 = sub_30 = None
    mul_106: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_128, mul_79);  mul_79 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_91: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_129: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_91, [1584, 768])
    mm_8: "f32[1584, 768]" = torch.ops.aten.mm.default(view_129, permute_91);  permute_91 = None
    permute_92: "f32[768, 1584]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_92, view_115);  permute_92 = view_115 = None
    permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[768]" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    view_131: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 198, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_132: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_131, [8, 198, 12, 64]);  view_131 = None
    permute_95: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_95, getitem_123, getitem_124, getitem_125, None, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False]);  permute_95 = getitem_123 = getitem_124 = getitem_125 = alias_12 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_134: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_135: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_136: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_134, getitem_135, getitem_136]);  getitem_134 = getitem_135 = getitem_136 = None
    view_133: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_1, [3, 8, 12, 198, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_96: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_133, [1, 3, 0, 2, 4]);  view_133 = None
    clone_37: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_134: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_37, [8, 198, 2304]);  clone_37 = None
    view_135: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_134, [1584, 2304]);  view_134 = None
    mm_10: "f32[1584, 768]" = torch.ops.aten.mm.default(view_135, permute_97);  permute_97 = None
    permute_98: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_11: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_98, view_111);  permute_98 = view_111 = None
    permute_99: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_14: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[2304]" = torch.ops.aten.reshape.default(sum_14, [2304]);  sum_14 = None
    permute_100: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    view_137: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_10, [8, 198, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_108: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_137, primals_138);  primals_138 = None
    mul_109: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_15: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_77);  mul_108 = None
    sum_16: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_16);  sum_16 = None
    sub_32: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_15);  mul_109 = sum_15 = None
    sub_33: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
    mul_112: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_33);  div_4 = sub_33 = None
    mul_113: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_137, mul_77);  mul_77 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_92: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_91, mul_112);  add_91 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_138: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_92, [1584, 768])
    mm_12: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_138, permute_101);  permute_101 = None
    permute_102: "f32[768, 1584]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_102, view_109);  permute_102 = view_109 = None
    permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
    view_139: "f32[768]" = torch.ops.aten.reshape.default(sum_19, [768]);  sum_19 = None
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    view_140: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_12, [8, 198, 3072]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_115: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
    mul_116: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, view_108)
    mul_117: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
    exp_1: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
    mul_118: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_119: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, mul_118);  view_108 = mul_118 = None
    add_94: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
    mul_120: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_140, add_94);  view_140 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_120, [1584, 3072]);  mul_120 = None
    mm_14: "f32[1584, 768]" = torch.ops.aten.mm.default(view_141, permute_105);  permute_105 = None
    permute_106: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_106, view_107);  permute_106 = view_107 = None
    permute_107: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[3072]" = torch.ops.aten.reshape.default(sum_20, [3072]);  sum_20 = None
    permute_108: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_143: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 198, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_122: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_143, primals_132);  primals_132 = None
    mul_123: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_21: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_72);  mul_122 = None
    sum_22: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_22);  sum_22 = None
    sub_35: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_21);  mul_123 = sum_21 = None
    sub_36: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
    mul_126: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_36);  div_5 = sub_36 = None
    mul_127: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_143, mul_72);  mul_72 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_95: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_92, mul_126);  add_92 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_144: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_95, [1584, 768])
    mm_16: "f32[1584, 768]" = torch.ops.aten.mm.default(view_144, permute_109);  permute_109 = None
    permute_110: "f32[768, 1584]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_17: "f32[768, 768]" = torch.ops.aten.mm.default(permute_110, view_105);  permute_110 = view_105 = None
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[768]" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_146: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_16, [8, 198, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_147: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_146, [8, 198, 12, 64]);  view_146 = None
    permute_113: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_113, getitem_112, getitem_113, getitem_114, None, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False]);  permute_113 = getitem_112 = getitem_113 = getitem_114 = alias_13 = getitem_116 = getitem_117 = getitem_118 = None
    getitem_138: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_139: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_140: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140]);  getitem_138 = getitem_139 = getitem_140 = None
    view_148: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_2, [3, 8, 12, 198, 64]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_114: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_148, [1, 3, 0, 2, 4]);  view_148 = None
    clone_38: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_149: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_38, [8, 198, 2304]);  clone_38 = None
    view_150: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_149, [1584, 2304]);  view_149 = None
    mm_18: "f32[1584, 768]" = torch.ops.aten.mm.default(view_150, permute_115);  permute_115 = None
    permute_116: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_150, [1, 0])
    mm_19: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_116, view_101);  permute_116 = view_101 = None
    permute_117: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_26: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
    view_151: "f32[2304]" = torch.ops.aten.reshape.default(sum_26, [2304]);  sum_26 = None
    permute_118: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    view_152: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 198, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_129: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_152, primals_126);  primals_126 = None
    mul_130: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_129, 768)
    sum_27: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_70);  mul_129 = None
    sum_28: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_28);  sum_28 = None
    sub_38: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_130, sum_27);  mul_130 = sum_27 = None
    sub_39: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
    mul_133: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_39);  div_6 = sub_39 = None
    mul_134: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_70);  mul_70 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_96: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_95, mul_133);  add_95 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_153: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_96, [1584, 768])
    mm_20: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_153, permute_119);  permute_119 = None
    permute_120: "f32[768, 1584]" = torch.ops.aten.permute.default(view_153, [1, 0])
    mm_21: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_120, view_99);  permute_120 = view_99 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
    view_154: "f32[768]" = torch.ops.aten.reshape.default(sum_31, [768]);  sum_31 = None
    permute_122: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    view_155: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_20, [8, 198, 3072]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_137: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_138: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
    exp_2: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
    mul_139: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_140: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_139);  view_98 = mul_139 = None
    add_98: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
    mul_141: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_155, add_98);  view_155 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_156: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_141, [1584, 3072]);  mul_141 = None
    mm_22: "f32[1584, 768]" = torch.ops.aten.mm.default(view_156, permute_123);  permute_123 = None
    permute_124: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_23: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_124, view_97);  permute_124 = view_97 = None
    permute_125: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_32: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
    view_157: "f32[3072]" = torch.ops.aten.reshape.default(sum_32, [3072]);  sum_32 = None
    permute_126: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_158: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 198, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_143: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_158, primals_120);  primals_120 = None
    mul_144: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_33: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_65);  mul_143 = None
    sum_34: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_34);  sum_34 = None
    sub_41: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_33);  mul_144 = sum_33 = None
    sub_42: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
    mul_147: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_42);  div_7 = sub_42 = None
    mul_148: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_158, mul_65);  mul_65 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_99: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_96, mul_147);  add_96 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_159: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_99, [1584, 768])
    mm_24: "f32[1584, 768]" = torch.ops.aten.mm.default(view_159, permute_127);  permute_127 = None
    permute_128: "f32[768, 1584]" = torch.ops.aten.permute.default(view_159, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_128, view_95);  permute_128 = view_95 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
    view_160: "f32[768]" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    view_161: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_24, [8, 198, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_162: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_161, [8, 198, 12, 64]);  view_161 = None
    permute_131: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_131, getitem_101, getitem_102, getitem_103, None, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False]);  permute_131 = getitem_101 = getitem_102 = getitem_103 = alias_14 = getitem_105 = getitem_106 = getitem_107 = None
    getitem_142: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_143: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_144: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_142, getitem_143, getitem_144]);  getitem_142 = getitem_143 = getitem_144 = None
    view_163: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_3, [3, 8, 12, 198, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_132: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_163, [1, 3, 0, 2, 4]);  view_163 = None
    clone_39: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_164: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_39, [8, 198, 2304]);  clone_39 = None
    view_165: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_164, [1584, 2304]);  view_164 = None
    mm_26: "f32[1584, 768]" = torch.ops.aten.mm.default(view_165, permute_133);  permute_133 = None
    permute_134: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_165, [1, 0])
    mm_27: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_134, view_91);  permute_134 = view_91 = None
    permute_135: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_38: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
    view_166: "f32[2304]" = torch.ops.aten.reshape.default(sum_38, [2304]);  sum_38 = None
    permute_136: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_167: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 198, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_150: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_167, primals_114);  primals_114 = None
    mul_151: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_39: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_63);  mul_150 = None
    sum_40: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_40);  sum_40 = None
    sub_44: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_39);  mul_151 = sum_39 = None
    sub_45: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
    mul_154: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_45);  div_8 = sub_45 = None
    mul_155: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_167, mul_63);  mul_63 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_167, [0, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_100: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_99, mul_154);  add_99 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_100, [1584, 768])
    mm_28: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_168, permute_137);  permute_137 = None
    permute_138: "f32[768, 1584]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_89);  permute_138 = view_89 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
    permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    view_170: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_28, [8, 198, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_157: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_158: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_159: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
    exp_3: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
    mul_160: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_161: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_160);  view_88 = mul_160 = None
    add_102: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
    mul_162: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_170, add_102);  view_170 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_171: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_162, [1584, 3072]);  mul_162 = None
    mm_30: "f32[1584, 768]" = torch.ops.aten.mm.default(view_171, permute_141);  permute_141 = None
    permute_142: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_87);  permute_142 = view_87 = None
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[3072]" = torch.ops.aten.reshape.default(sum_44, [3072]);  sum_44 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_173: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_30, [8, 198, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_164: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_173, primals_108);  primals_108 = None
    mul_165: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_45: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_58);  mul_164 = None
    sum_46: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_46);  sum_46 = None
    sub_47: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_45);  mul_165 = sum_45 = None
    sub_48: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
    mul_168: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_48);  div_9 = sub_48 = None
    mul_169: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_173, mul_58);  mul_58 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_173, [0, 1]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_103: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_100, mul_168);  add_100 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_174: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_103, [1584, 768])
    mm_32: "f32[1584, 768]" = torch.ops.aten.mm.default(view_174, permute_145);  permute_145 = None
    permute_146: "f32[768, 1584]" = torch.ops.aten.permute.default(view_174, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_85);  permute_146 = view_85 = None
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
    view_175: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
    permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_176: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_32, [8, 198, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_177: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_176, [8, 198, 12, 64]);  view_176 = None
    permute_149: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_149, getitem_90, getitem_91, getitem_92, None, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False]);  permute_149 = getitem_90 = getitem_91 = getitem_92 = alias_15 = getitem_94 = getitem_95 = getitem_96 = None
    getitem_146: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_147: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_148: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
    view_178: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_4, [3, 8, 12, 198, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_150: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_178, [1, 3, 0, 2, 4]);  view_178 = None
    clone_40: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_179: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_40, [8, 198, 2304]);  clone_40 = None
    view_180: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_179, [1584, 2304]);  view_179 = None
    mm_34: "f32[1584, 768]" = torch.ops.aten.mm.default(view_180, permute_151);  permute_151 = None
    permute_152: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_35: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_152, view_81);  permute_152 = view_81 = None
    permute_153: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_50: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
    view_181: "f32[2304]" = torch.ops.aten.reshape.default(sum_50, [2304]);  sum_50 = None
    permute_154: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_182: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_34, [8, 198, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_171: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_182, primals_102);  primals_102 = None
    mul_172: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
    sum_51: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_56);  mul_171 = None
    sum_52: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_52);  sum_52 = None
    sub_50: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_51);  mul_172 = sum_51 = None
    sub_51: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
    mul_175: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_51);  div_10 = sub_51 = None
    mul_176: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_182, mul_56);  mul_56 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_182, [0, 1]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_104: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_103, mul_175);  add_103 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_104, [1584, 768])
    mm_36: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_183, permute_155);  permute_155 = None
    permute_156: "f32[768, 1584]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_156, view_79);  permute_156 = view_79 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[768]" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
    permute_158: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_185: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_36, [8, 198, 3072]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_178: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_179: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_180: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
    exp_4: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
    mul_181: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_182: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_181);  view_78 = mul_181 = None
    add_106: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
    mul_183: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_185, add_106);  view_185 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_183, [1584, 3072]);  mul_183 = None
    mm_38: "f32[1584, 768]" = torch.ops.aten.mm.default(view_186, permute_159);  permute_159 = None
    permute_160: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_160, view_77);  permute_160 = view_77 = None
    permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_56: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[3072]" = torch.ops.aten.reshape.default(sum_56, [3072]);  sum_56 = None
    permute_162: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_188: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 198, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_185: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_188, primals_96);  primals_96 = None
    mul_186: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
    sum_57: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
    mul_187: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_51);  mul_185 = None
    sum_58: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
    mul_188: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_58);  sum_58 = None
    sub_53: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_57);  mul_186 = sum_57 = None
    sub_54: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
    mul_189: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_54);  div_11 = sub_54 = None
    mul_190: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_51);  mul_51 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_107: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_104, mul_189);  add_104 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_189: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_107, [1584, 768])
    mm_40: "f32[1584, 768]" = torch.ops.aten.mm.default(view_189, permute_163);  permute_163 = None
    permute_164: "f32[768, 1584]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_41: "f32[768, 768]" = torch.ops.aten.mm.default(permute_164, view_75);  permute_164 = view_75 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
    view_190: "f32[768]" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_191: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_40, [8, 198, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_192: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_191, [8, 198, 12, 64]);  view_191 = None
    permute_167: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, getitem_79, getitem_80, getitem_81, None, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_167 = getitem_79 = getitem_80 = getitem_81 = alias_16 = getitem_83 = getitem_84 = getitem_85 = None
    getitem_150: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_151: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_152: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
    view_193: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_5, [3, 8, 12, 198, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_168: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_193, [1, 3, 0, 2, 4]);  view_193 = None
    clone_41: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_194: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_41, [8, 198, 2304]);  clone_41 = None
    view_195: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_194, [1584, 2304]);  view_194 = None
    mm_42: "f32[1584, 768]" = torch.ops.aten.mm.default(view_195, permute_169);  permute_169 = None
    permute_170: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_43: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_170, view_71);  permute_170 = view_71 = None
    permute_171: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_62: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
    view_196: "f32[2304]" = torch.ops.aten.reshape.default(sum_62, [2304]);  sum_62 = None
    permute_172: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    view_197: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_42, [8, 198, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_192: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_197, primals_90);  primals_90 = None
    mul_193: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_63: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_49);  mul_192 = None
    sum_64: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_64);  sum_64 = None
    sub_56: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_63);  mul_193 = sum_63 = None
    sub_57: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
    mul_196: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_57);  div_12 = sub_57 = None
    mul_197: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_197, mul_49);  mul_49 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_108: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_107, mul_196);  add_107 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_108, [1584, 768])
    mm_44: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_198, permute_173);  permute_173 = None
    permute_174: "f32[768, 1584]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_45: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_174, view_69);  permute_174 = view_69 = None
    permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    view_200: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_44, [8, 198, 3072]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_199: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_200: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_201: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
    exp_5: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
    mul_202: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_203: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, mul_202);  view_68 = mul_202 = None
    add_110: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
    mul_204: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_200, add_110);  view_200 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_201: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_204, [1584, 3072]);  mul_204 = None
    mm_46: "f32[1584, 768]" = torch.ops.aten.mm.default(view_201, permute_177);  permute_177 = None
    permute_178: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_47: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_178, view_67);  permute_178 = view_67 = None
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[3072]" = torch.ops.aten.reshape.default(sum_68, [3072]);  sum_68 = None
    permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    view_203: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 198, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_206: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_203, primals_84);  primals_84 = None
    mul_207: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
    sum_69: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
    mul_208: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_44);  mul_206 = None
    sum_70: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
    mul_209: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_70);  sum_70 = None
    sub_59: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_69);  mul_207 = sum_69 = None
    sub_60: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
    mul_210: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_60);  div_13 = sub_60 = None
    mul_211: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_203, mul_44);  mul_44 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_111: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_108, mul_210);  add_108 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_204: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_111, [1584, 768])
    mm_48: "f32[1584, 768]" = torch.ops.aten.mm.default(view_204, permute_181);  permute_181 = None
    permute_182: "f32[768, 1584]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_182, view_65);  permute_182 = view_65 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    view_206: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 198, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_207: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_206, [8, 198, 12, 64]);  view_206 = None
    permute_185: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_185, getitem_68, getitem_69, getitem_70, None, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False]);  permute_185 = getitem_68 = getitem_69 = getitem_70 = alias_17 = getitem_72 = getitem_73 = getitem_74 = None
    getitem_154: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_155: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_156: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
    view_208: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_6, [3, 8, 12, 198, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_186: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_208, [1, 3, 0, 2, 4]);  view_208 = None
    clone_42: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_209: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_42, [8, 198, 2304]);  clone_42 = None
    view_210: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_209, [1584, 2304]);  view_209 = None
    mm_50: "f32[1584, 768]" = torch.ops.aten.mm.default(view_210, permute_187);  permute_187 = None
    permute_188: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_210, [1, 0])
    mm_51: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_188, view_61);  permute_188 = view_61 = None
    permute_189: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_74: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[2304]" = torch.ops.aten.reshape.default(sum_74, [2304]);  sum_74 = None
    permute_190: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_212: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_50, [8, 198, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_213: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_212, primals_78);  primals_78 = None
    mul_214: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_75: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_42);  mul_213 = None
    sum_76: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_76);  sum_76 = None
    sub_62: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_75);  mul_214 = sum_75 = None
    sub_63: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
    mul_217: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_63);  div_14 = sub_63 = None
    mul_218: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_42);  mul_42 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_112: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_111, mul_217);  add_111 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_213: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_112, [1584, 768])
    mm_52: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_213, permute_191);  permute_191 = None
    permute_192: "f32[768, 1584]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_192, view_59);  permute_192 = view_59 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_215: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_52, [8, 198, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_220: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_221: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_222: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_6: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_224: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_223);  view_58 = mul_223 = None
    add_114: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_215, add_114);  view_215 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_216: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_225, [1584, 3072]);  mul_225 = None
    mm_54: "f32[1584, 768]" = torch.ops.aten.mm.default(view_216, permute_195);  permute_195 = None
    permute_196: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_196, view_57);  permute_196 = view_57 = None
    permute_197: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
    view_217: "f32[3072]" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
    permute_198: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_218: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 198, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_227: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_218, primals_72);  primals_72 = None
    mul_228: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
    sum_81: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_37);  mul_227 = None
    sum_82: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_82);  sum_82 = None
    sub_65: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_81);  mul_228 = sum_81 = None
    sub_66: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
    mul_231: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_66);  div_15 = sub_66 = None
    mul_232: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_218, mul_37);  mul_37 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_218, [0, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_115: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_112, mul_231);  add_112 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_219: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_115, [1584, 768])
    mm_56: "f32[1584, 768]" = torch.ops.aten.mm.default(view_219, permute_199);  permute_199 = None
    permute_200: "f32[768, 1584]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_55);  permute_200 = view_55 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_221: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_56, [8, 198, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_222: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_221, [8, 198, 12, 64]);  view_221 = None
    permute_203: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_203, getitem_57, getitem_58, getitem_59, None, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_203 = getitem_57 = getitem_58 = getitem_59 = alias_18 = getitem_61 = getitem_62 = getitem_63 = None
    getitem_158: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_159: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_160: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
    view_223: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_7, [3, 8, 12, 198, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_204: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_223, [1, 3, 0, 2, 4]);  view_223 = None
    clone_43: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_224: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_43, [8, 198, 2304]);  clone_43 = None
    view_225: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_224, [1584, 2304]);  view_224 = None
    mm_58: "f32[1584, 768]" = torch.ops.aten.mm.default(view_225, permute_205);  permute_205 = None
    permute_206: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_225, [1, 0])
    mm_59: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_206, view_51);  permute_206 = view_51 = None
    permute_207: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_86: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
    view_226: "f32[2304]" = torch.ops.aten.reshape.default(sum_86, [2304]);  sum_86 = None
    permute_208: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_227: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_58, [8, 198, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_234: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_227, primals_66);  primals_66 = None
    mul_235: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_234, 768)
    sum_87: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_234, mul_35);  mul_234 = None
    sum_88: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_88);  sum_88 = None
    sub_68: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_235, sum_87);  mul_235 = sum_87 = None
    sub_69: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
    mul_238: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_69);  div_16 = sub_69 = None
    mul_239: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_227, mul_35);  mul_35 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_227, [0, 1]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_116: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_115, mul_238);  add_115 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_228: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_116, [1584, 768])
    mm_60: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_228, permute_209);  permute_209 = None
    permute_210: "f32[768, 1584]" = torch.ops.aten.permute.default(view_228, [1, 0])
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_49);  permute_210 = view_49 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
    view_229: "f32[768]" = torch.ops.aten.reshape.default(sum_91, [768]);  sum_91 = None
    permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_230: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_60, [8, 198, 3072]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_241: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_242: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, view_48)
    mul_243: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_7: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_245: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, mul_244);  view_48 = mul_244 = None
    add_118: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_230, add_118);  view_230 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_246, [1584, 3072]);  mul_246 = None
    mm_62: "f32[1584, 768]" = torch.ops.aten.mm.default(view_231, permute_213);  permute_213 = None
    permute_214: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_47);  permute_214 = view_47 = None
    permute_215: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_92: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[3072]" = torch.ops.aten.reshape.default(sum_92, [3072]);  sum_92 = None
    permute_216: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_233: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_62, [8, 198, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_248: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_233, primals_60);  primals_60 = None
    mul_249: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_93: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_30);  mul_248 = None
    sum_94: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_94);  sum_94 = None
    sub_71: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_93);  mul_249 = sum_93 = None
    sub_72: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
    mul_252: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_72);  div_17 = sub_72 = None
    mul_253: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_30);  mul_30 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_119: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_116, mul_252);  add_116 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_234: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_119, [1584, 768])
    mm_64: "f32[1584, 768]" = torch.ops.aten.mm.default(view_234, permute_217);  permute_217 = None
    permute_218: "f32[768, 1584]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_65: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_45);  permute_218 = view_45 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_236: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_64, [8, 198, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_237: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_236, [8, 198, 12, 64]);  view_236 = None
    permute_221: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_221, getitem_46, getitem_47, getitem_48, None, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False]);  permute_221 = getitem_46 = getitem_47 = getitem_48 = alias_19 = getitem_50 = getitem_51 = getitem_52 = None
    getitem_162: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_163: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_164: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
    view_238: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_8, [3, 8, 12, 198, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_222: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_238, [1, 3, 0, 2, 4]);  view_238 = None
    clone_44: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_239: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_44, [8, 198, 2304]);  clone_44 = None
    view_240: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_239, [1584, 2304]);  view_239 = None
    mm_66: "f32[1584, 768]" = torch.ops.aten.mm.default(view_240, permute_223);  permute_223 = None
    permute_224: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_67: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_224, view_41);  permute_224 = view_41 = None
    permute_225: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_98: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
    view_241: "f32[2304]" = torch.ops.aten.reshape.default(sum_98, [2304]);  sum_98 = None
    permute_226: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_242: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_66, [8, 198, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_255: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_242, primals_54);  primals_54 = None
    mul_256: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_99: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_28);  mul_255 = None
    sum_100: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_100);  sum_100 = None
    sub_74: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_99);  mul_256 = sum_99 = None
    sub_75: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
    mul_259: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_75);  div_18 = sub_75 = None
    mul_260: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_242, mul_28);  mul_28 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_120: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_119, mul_259);  add_119 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_243: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_120, [1584, 768])
    mm_68: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_243, permute_227);  permute_227 = None
    permute_228: "f32[768, 1584]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_69: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_228, view_39);  permute_228 = view_39 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
    permute_230: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_245: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_68, [8, 198, 3072]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_262: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_263: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_264: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_8: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_266: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_265);  view_38 = mul_265 = None
    add_122: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_245, add_122);  view_245 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_267, [1584, 3072]);  mul_267 = None
    mm_70: "f32[1584, 768]" = torch.ops.aten.mm.default(view_246, permute_231);  permute_231 = None
    permute_232: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_71: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_232, view_37);  permute_232 = view_37 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[3072]" = torch.ops.aten.reshape.default(sum_104, [3072]);  sum_104 = None
    permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_248: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_70, [8, 198, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_269: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_248, primals_48);  primals_48 = None
    mul_270: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_105: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_23);  mul_269 = None
    sum_106: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_106);  sum_106 = None
    sub_77: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_105);  mul_270 = sum_105 = None
    sub_78: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
    mul_273: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_78);  div_19 = sub_78 = None
    mul_274: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_248, mul_23);  mul_23 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_120, mul_273);  add_120 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_249: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_123, [1584, 768])
    mm_72: "f32[1584, 768]" = torch.ops.aten.mm.default(view_249, permute_235);  permute_235 = None
    permute_236: "f32[768, 1584]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_236, view_35);  permute_236 = view_35 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[768]" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    view_251: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_72, [8, 198, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_252: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_251, [8, 198, 12, 64]);  view_251 = None
    permute_239: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_239, getitem_35, getitem_36, getitem_37, None, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False]);  permute_239 = getitem_35 = getitem_36 = getitem_37 = alias_20 = getitem_39 = getitem_40 = getitem_41 = None
    getitem_166: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_167: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_168: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
    view_253: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_9, [3, 8, 12, 198, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_240: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_253, [1, 3, 0, 2, 4]);  view_253 = None
    clone_45: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_254: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_45, [8, 198, 2304]);  clone_45 = None
    view_255: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_254, [1584, 2304]);  view_254 = None
    mm_74: "f32[1584, 768]" = torch.ops.aten.mm.default(view_255, permute_241);  permute_241 = None
    permute_242: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_255, [1, 0])
    mm_75: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_242, view_31);  permute_242 = view_31 = None
    permute_243: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_110: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
    view_256: "f32[2304]" = torch.ops.aten.reshape.default(sum_110, [2304]);  sum_110 = None
    permute_244: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_257: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_74, [8, 198, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_276: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_257, primals_42);  primals_42 = None
    mul_277: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
    sum_111: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
    mul_278: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_21);  mul_276 = None
    sum_112: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
    mul_279: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_112);  sum_112 = None
    sub_80: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_111);  mul_277 = sum_111 = None
    sub_81: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
    mul_280: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_81);  div_20 = sub_81 = None
    mul_281: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_257, mul_21);  mul_21 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_257, [0, 1]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_124: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_123, mul_280);  add_123 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_258: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_124, [1584, 768])
    mm_76: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_258, permute_245);  permute_245 = None
    permute_246: "f32[768, 1584]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_246, view_29);  permute_246 = view_29 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[768]" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_260: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_76, [8, 198, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_283: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_284: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, view_28)
    mul_285: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
    exp_9: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
    mul_286: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_287: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, mul_286);  view_28 = mul_286 = None
    add_126: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
    mul_288: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_260, add_126);  view_260 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_261: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_288, [1584, 3072]);  mul_288 = None
    mm_78: "f32[1584, 768]" = torch.ops.aten.mm.default(view_261, permute_249);  permute_249 = None
    permute_250: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_250, view_27);  permute_250 = view_27 = None
    permute_251: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[3072]" = torch.ops.aten.reshape.default(sum_116, [3072]);  sum_116 = None
    permute_252: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_263: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 198, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_290: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_263, primals_36);  primals_36 = None
    mul_291: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_290, 768)
    sum_117: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
    mul_292: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_16);  mul_290 = None
    sum_118: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
    mul_293: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_118);  sum_118 = None
    sub_83: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_291, sum_117);  mul_291 = sum_117 = None
    sub_84: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
    mul_294: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_84);  div_21 = sub_84 = None
    mul_295: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_16);  mul_16 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_127: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_124, mul_294);  add_124 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_264: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_127, [1584, 768])
    mm_80: "f32[1584, 768]" = torch.ops.aten.mm.default(view_264, permute_253);  permute_253 = None
    permute_254: "f32[768, 1584]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_254, view_25);  permute_254 = view_25 = None
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[768]" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    view_266: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_80, [8, 198, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_267: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_266, [8, 198, 12, 64]);  view_266 = None
    permute_257: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_257, getitem_24, getitem_25, getitem_26, None, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False]);  permute_257 = getitem_24 = getitem_25 = getitem_26 = alias_21 = getitem_28 = getitem_29 = getitem_30 = None
    getitem_170: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
    getitem_171: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
    getitem_172: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
    view_268: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_10, [3, 8, 12, 198, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_258: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_268, [1, 3, 0, 2, 4]);  view_268 = None
    clone_46: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_269: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_46, [8, 198, 2304]);  clone_46 = None
    view_270: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_269, [1584, 2304]);  view_269 = None
    mm_82: "f32[1584, 768]" = torch.ops.aten.mm.default(view_270, permute_259);  permute_259 = None
    permute_260: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_83: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_260, view_21);  permute_260 = view_21 = None
    permute_261: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_122: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[2304]" = torch.ops.aten.reshape.default(sum_122, [2304]);  sum_122 = None
    permute_262: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    view_272: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 198, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_297: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_272, primals_30);  primals_30 = None
    mul_298: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_123: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_14);  mul_297 = None
    sum_124: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_124);  sum_124 = None
    sub_86: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_123);  mul_298 = sum_123 = None
    sub_87: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
    mul_301: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_87);  div_22 = sub_87 = None
    mul_302: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_272, mul_14);  mul_14 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_128: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_127, mul_301);  add_127 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_128, [1584, 768])
    mm_84: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_273, permute_263);  permute_263 = None
    permute_264: "f32[768, 1584]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_264, view_19);  permute_264 = view_19 = None
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    permute_266: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_275: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_84, [8, 198, 3072]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_304: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_305: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_306: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_10: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_308: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_307);  view_18 = mul_307 = None
    add_130: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_275, add_130);  view_275 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_309, [1584, 3072]);  mul_309 = None
    mm_86: "f32[1584, 768]" = torch.ops.aten.mm.default(view_276, permute_267);  permute_267 = None
    permute_268: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_268, view_17);  permute_268 = view_17 = None
    permute_269: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_128: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[3072]" = torch.ops.aten.reshape.default(sum_128, [3072]);  sum_128 = None
    permute_270: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_278: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_86, [8, 198, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_311: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_278, primals_24);  primals_24 = None
    mul_312: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_129: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_9);  mul_311 = None
    sum_130: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_130);  sum_130 = None
    sub_89: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_129);  mul_312 = sum_129 = None
    sub_90: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
    mul_315: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_90);  div_23 = sub_90 = None
    mul_316: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_278, mul_9);  mul_9 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_131: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_128, mul_315);  add_128 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_279: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_131, [1584, 768])
    mm_88: "f32[1584, 768]" = torch.ops.aten.mm.default(view_279, permute_271);  permute_271 = None
    permute_272: "f32[768, 1584]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_15);  permute_272 = view_15 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[768]" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    view_281: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_88, [8, 198, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_282: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_281, [8, 198, 12, 64]);  view_281 = None
    permute_275: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_275, getitem_13, getitem_14, getitem_15, None, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False]);  permute_275 = getitem_13 = getitem_14 = getitem_15 = alias_22 = getitem_17 = getitem_18 = getitem_19 = None
    getitem_174: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
    getitem_175: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
    getitem_176: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
    view_283: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_11, [3, 8, 12, 198, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_276: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_283, [1, 3, 0, 2, 4]);  view_283 = None
    clone_47: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    view_284: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_47, [8, 198, 2304]);  clone_47 = None
    view_285: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_284, [1584, 2304]);  view_284 = None
    mm_90: "f32[1584, 768]" = torch.ops.aten.mm.default(view_285, permute_277);  permute_277 = None
    permute_278: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_91: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_278, view_11);  permute_278 = view_11 = None
    permute_279: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_134: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[2304]" = torch.ops.aten.reshape.default(sum_134, [2304]);  sum_134 = None
    permute_280: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_287: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_90, [8, 198, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_318: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_287, primals_18);  primals_18 = None
    mul_319: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
    sum_135: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_318, mul_7);  mul_318 = None
    sum_136: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_136);  sum_136 = None
    sub_92: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_135);  mul_319 = sum_135 = None
    sub_93: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
    mul_322: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_93);  div_24 = sub_93 = None
    mul_323: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_287, mul_7);  mul_7 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_132: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_131, mul_322);  add_131 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_288: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_132, [1584, 768])
    mm_92: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_288, permute_281);  permute_281 = None
    permute_282: "f32[768, 1584]" = torch.ops.aten.permute.default(view_288, [1, 0])
    mm_93: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_282, view_9);  permute_282 = view_9 = None
    permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
    view_289: "f32[768]" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
    permute_284: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    view_290: "f32[8, 198, 3072]" = torch.ops.aten.reshape.default(mm_92, [8, 198, 3072]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_325: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_326: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_327: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_11: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_329: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, mul_328);  view_8 = mul_328 = None
    add_134: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_290, add_134);  view_290 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_291: "f32[1584, 3072]" = torch.ops.aten.reshape.default(mul_330, [1584, 3072]);  mul_330 = None
    mm_94: "f32[1584, 768]" = torch.ops.aten.mm.default(view_291, permute_285);  permute_285 = None
    permute_286: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_95: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_286, view_7);  permute_286 = view_7 = None
    permute_287: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[3072]" = torch.ops.aten.reshape.default(sum_140, [3072]);  sum_140 = None
    permute_288: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    view_293: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 198, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_332: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_293, primals_12);  primals_12 = None
    mul_333: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
    sum_141: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_2);  mul_332 = None
    sum_142: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_142);  sum_142 = None
    sub_95: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_141);  mul_333 = sum_141 = None
    sub_96: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
    mul_336: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_96);  div_25 = sub_96 = None
    mul_337: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_293, mul_2);  mul_2 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_293, [0, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_135: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_132, mul_336);  add_132 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_294: "f32[1584, 768]" = torch.ops.aten.reshape.default(add_135, [1584, 768])
    mm_96: "f32[1584, 768]" = torch.ops.aten.mm.default(view_294, permute_289);  permute_289 = None
    permute_290: "f32[768, 1584]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_5);  permute_290 = view_5 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_145: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
    permute_292: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    view_296: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_96, [8, 198, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_297: "f32[8, 198, 12, 64]" = torch.ops.aten.reshape.default(view_296, [8, 198, 12, 64]);  view_296 = None
    permute_293: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_293, getitem_2, getitem_3, getitem_4, None, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False]);  permute_293 = getitem_2 = getitem_3 = getitem_4 = alias_23 = getitem_6 = getitem_7 = getitem_8 = None
    getitem_178: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
    getitem_179: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
    getitem_180: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
    view_298: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.reshape.default(cat_12, [3, 8, 12, 198, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_294: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_298, [1, 3, 0, 2, 4]);  view_298 = None
    clone_48: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_299: "f32[8, 198, 2304]" = torch.ops.aten.reshape.default(clone_48, [8, 198, 2304]);  clone_48 = None
    view_300: "f32[1584, 2304]" = torch.ops.aten.reshape.default(view_299, [1584, 2304]);  view_299 = None
    mm_98: "f32[1584, 768]" = torch.ops.aten.mm.default(view_300, permute_295);  permute_295 = None
    permute_296: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_99: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_296, view_1);  permute_296 = view_1 = None
    permute_297: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[2304]" = torch.ops.aten.reshape.default(sum_146, [2304]);  sum_146 = None
    permute_298: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_302: "f32[8, 198, 768]" = torch.ops.aten.reshape.default(mm_98, [8, 198, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_339: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_302, primals_6);  primals_6 = None
    mul_340: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_339, 768)
    sum_147: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul);  mul_339 = None
    sum_148: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul, sum_148);  sum_148 = None
    sub_98: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_340, sum_147);  mul_340 = sum_147 = None
    sub_99: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
    mul_343: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_99);  div_26 = sub_99 = None
    mul_344: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_302, mul);  mul = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_302, [0, 1]);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_136: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_135, mul_343);  add_135 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:104, code: x = x + pos_embed
    sum_151: "f32[1, 198, 768]" = torch.ops.aten.sum.dim_IntList(add_136, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:99, code: x = torch.cat((
    slice_3: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 0, 1)
    slice_4: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 1, 2)
    slice_5: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 2, 198);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:101, code: self.dist_token.expand(x.shape[0], -1, -1),
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_4, [0], True);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:100, code: self.cls_token.expand(x.shape[0], -1, -1),
    sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_3, [0], True);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_299: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_5, [0, 2, 1]);  slice_5 = None
    view_303: "f32[8, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_299, [8, 768, 14, 14]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_303, primals_156, primals_4, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_303 = primals_156 = primals_4 = None
    getitem_183: "f32[768, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return [sum_151, sum_153, sum_152, getitem_183, sum_154, sum_149, sum_150, permute_298, view_301, permute_292, view_295, sum_143, sum_144, permute_288, view_292, permute_284, view_289, sum_137, sum_138, permute_280, view_286, permute_274, view_280, sum_131, sum_132, permute_270, view_277, permute_266, view_274, sum_125, sum_126, permute_262, view_271, permute_256, view_265, sum_119, sum_120, permute_252, view_262, permute_248, view_259, sum_113, sum_114, permute_244, view_256, permute_238, view_250, sum_107, sum_108, permute_234, view_247, permute_230, view_244, sum_101, sum_102, permute_226, view_241, permute_220, view_235, sum_95, sum_96, permute_216, view_232, permute_212, view_229, sum_89, sum_90, permute_208, view_226, permute_202, view_220, sum_83, sum_84, permute_198, view_217, permute_194, view_214, sum_77, sum_78, permute_190, view_211, permute_184, view_205, sum_71, sum_72, permute_180, view_202, permute_176, view_199, sum_65, sum_66, permute_172, view_196, permute_166, view_190, sum_59, sum_60, permute_162, view_187, permute_158, view_184, sum_53, sum_54, permute_154, view_181, permute_148, view_175, sum_47, sum_48, permute_144, view_172, permute_140, view_169, sum_41, sum_42, permute_136, view_166, permute_130, view_160, sum_35, sum_36, permute_126, view_157, permute_122, view_154, sum_29, sum_30, permute_118, view_151, permute_112, view_145, sum_23, sum_24, permute_108, view_142, permute_104, view_139, sum_17, sum_18, permute_100, view_136, permute_94, view_130, sum_11, sum_12, permute_90, view_127, permute_86, view_124, sum_5, sum_6, permute_82, view_121, permute_78, view_121, None]
    