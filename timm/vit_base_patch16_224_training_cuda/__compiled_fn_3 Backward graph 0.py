from __future__ import annotations



def forward(self, primals_3: "f32[768, 3, 16, 16]", primals_5: "f32[768]", primals_11: "f32[768]", primals_17: "f32[768]", primals_23: "f32[768]", primals_29: "f32[768]", primals_35: "f32[768]", primals_41: "f32[768]", primals_47: "f32[768]", primals_53: "f32[768]", primals_59: "f32[768]", primals_65: "f32[768]", primals_71: "f32[768]", primals_77: "f32[768]", primals_83: "f32[768]", primals_89: "f32[768]", primals_95: "f32[768]", primals_101: "f32[768]", primals_107: "f32[768]", primals_113: "f32[768]", primals_119: "f32[768]", primals_125: "f32[768]", primals_131: "f32[768]", primals_137: "f32[768]", primals_143: "f32[768]", primals_149: "f32[768]", primals_153: "f32[8, 3, 224, 224]", mul: "f32[8, 197, 768]", view_1: "f32[1576, 768]", getitem_2: "f32[8, 12, 197, 64]", getitem_3: "f32[8, 12, 197, 64]", getitem_4: "f32[8, 12, 197, 64]", getitem_6: "f32[8, 12, 224]", getitem_7: "i64[]", getitem_8: "i64[]", view_5: "f32[1576, 768]", mul_2: "f32[8, 197, 768]", view_7: "f32[1576, 768]", addmm_2: "f32[1576, 3072]", view_9: "f32[1576, 3072]", mul_7: "f32[8, 197, 768]", view_11: "f32[1576, 768]", getitem_13: "f32[8, 12, 197, 64]", getitem_14: "f32[8, 12, 197, 64]", getitem_15: "f32[8, 12, 197, 64]", getitem_17: "f32[8, 12, 224]", getitem_18: "i64[]", getitem_19: "i64[]", view_15: "f32[1576, 768]", mul_9: "f32[8, 197, 768]", view_17: "f32[1576, 768]", addmm_6: "f32[1576, 3072]", view_19: "f32[1576, 3072]", mul_14: "f32[8, 197, 768]", view_21: "f32[1576, 768]", getitem_24: "f32[8, 12, 197, 64]", getitem_25: "f32[8, 12, 197, 64]", getitem_26: "f32[8, 12, 197, 64]", getitem_28: "f32[8, 12, 224]", getitem_29: "i64[]", getitem_30: "i64[]", view_25: "f32[1576, 768]", mul_16: "f32[8, 197, 768]", view_27: "f32[1576, 768]", addmm_10: "f32[1576, 3072]", view_29: "f32[1576, 3072]", mul_21: "f32[8, 197, 768]", view_31: "f32[1576, 768]", getitem_35: "f32[8, 12, 197, 64]", getitem_36: "f32[8, 12, 197, 64]", getitem_37: "f32[8, 12, 197, 64]", getitem_39: "f32[8, 12, 224]", getitem_40: "i64[]", getitem_41: "i64[]", view_35: "f32[1576, 768]", mul_23: "f32[8, 197, 768]", view_37: "f32[1576, 768]", addmm_14: "f32[1576, 3072]", view_39: "f32[1576, 3072]", mul_28: "f32[8, 197, 768]", view_41: "f32[1576, 768]", getitem_46: "f32[8, 12, 197, 64]", getitem_47: "f32[8, 12, 197, 64]", getitem_48: "f32[8, 12, 197, 64]", getitem_50: "f32[8, 12, 224]", getitem_51: "i64[]", getitem_52: "i64[]", view_45: "f32[1576, 768]", mul_30: "f32[8, 197, 768]", view_47: "f32[1576, 768]", addmm_18: "f32[1576, 3072]", view_49: "f32[1576, 3072]", mul_35: "f32[8, 197, 768]", view_51: "f32[1576, 768]", getitem_57: "f32[8, 12, 197, 64]", getitem_58: "f32[8, 12, 197, 64]", getitem_59: "f32[8, 12, 197, 64]", getitem_61: "f32[8, 12, 224]", getitem_62: "i64[]", getitem_63: "i64[]", view_55: "f32[1576, 768]", mul_37: "f32[8, 197, 768]", view_57: "f32[1576, 768]", addmm_22: "f32[1576, 3072]", view_59: "f32[1576, 3072]", mul_42: "f32[8, 197, 768]", view_61: "f32[1576, 768]", getitem_68: "f32[8, 12, 197, 64]", getitem_69: "f32[8, 12, 197, 64]", getitem_70: "f32[8, 12, 197, 64]", getitem_72: "f32[8, 12, 224]", getitem_73: "i64[]", getitem_74: "i64[]", view_65: "f32[1576, 768]", mul_44: "f32[8, 197, 768]", view_67: "f32[1576, 768]", addmm_26: "f32[1576, 3072]", view_69: "f32[1576, 3072]", mul_49: "f32[8, 197, 768]", view_71: "f32[1576, 768]", getitem_79: "f32[8, 12, 197, 64]", getitem_80: "f32[8, 12, 197, 64]", getitem_81: "f32[8, 12, 197, 64]", getitem_83: "f32[8, 12, 224]", getitem_84: "i64[]", getitem_85: "i64[]", view_75: "f32[1576, 768]", mul_51: "f32[8, 197, 768]", view_77: "f32[1576, 768]", addmm_30: "f32[1576, 3072]", view_79: "f32[1576, 3072]", mul_56: "f32[8, 197, 768]", view_81: "f32[1576, 768]", getitem_90: "f32[8, 12, 197, 64]", getitem_91: "f32[8, 12, 197, 64]", getitem_92: "f32[8, 12, 197, 64]", getitem_94: "f32[8, 12, 224]", getitem_95: "i64[]", getitem_96: "i64[]", view_85: "f32[1576, 768]", mul_58: "f32[8, 197, 768]", view_87: "f32[1576, 768]", addmm_34: "f32[1576, 3072]", view_89: "f32[1576, 3072]", mul_63: "f32[8, 197, 768]", view_91: "f32[1576, 768]", getitem_101: "f32[8, 12, 197, 64]", getitem_102: "f32[8, 12, 197, 64]", getitem_103: "f32[8, 12, 197, 64]", getitem_105: "f32[8, 12, 224]", getitem_106: "i64[]", getitem_107: "i64[]", view_95: "f32[1576, 768]", mul_65: "f32[8, 197, 768]", view_97: "f32[1576, 768]", addmm_38: "f32[1576, 3072]", view_99: "f32[1576, 3072]", mul_70: "f32[8, 197, 768]", view_101: "f32[1576, 768]", getitem_112: "f32[8, 12, 197, 64]", getitem_113: "f32[8, 12, 197, 64]", getitem_114: "f32[8, 12, 197, 64]", getitem_116: "f32[8, 12, 224]", getitem_117: "i64[]", getitem_118: "i64[]", view_105: "f32[1576, 768]", mul_72: "f32[8, 197, 768]", view_107: "f32[1576, 768]", addmm_42: "f32[1576, 3072]", view_109: "f32[1576, 3072]", mul_77: "f32[8, 197, 768]", view_111: "f32[1576, 768]", getitem_123: "f32[8, 12, 197, 64]", getitem_124: "f32[8, 12, 197, 64]", getitem_125: "f32[8, 12, 197, 64]", getitem_127: "f32[8, 12, 224]", getitem_128: "i64[]", getitem_129: "i64[]", view_115: "f32[1576, 768]", mul_79: "f32[8, 197, 768]", view_117: "f32[1576, 768]", addmm_46: "f32[1576, 3072]", view_119: "f32[1576, 3072]", mul_84: "f32[8, 197, 768]", clone_37: "f32[8, 768]", permute_74: "f32[1000, 768]", div: "f32[8, 197, 1]", permute_78: "f32[768, 3072]", permute_82: "f32[3072, 768]", div_1: "f32[8, 197, 1]", permute_86: "f32[768, 768]", alias_12: "f32[8, 12, 197, 64]", permute_92: "f32[2304, 768]", div_2: "f32[8, 197, 1]", permute_96: "f32[768, 3072]", permute_100: "f32[3072, 768]", div_3: "f32[8, 197, 1]", permute_104: "f32[768, 768]", alias_13: "f32[8, 12, 197, 64]", permute_110: "f32[2304, 768]", div_4: "f32[8, 197, 1]", permute_114: "f32[768, 3072]", permute_118: "f32[3072, 768]", div_5: "f32[8, 197, 1]", permute_122: "f32[768, 768]", alias_14: "f32[8, 12, 197, 64]", permute_128: "f32[2304, 768]", div_6: "f32[8, 197, 1]", permute_132: "f32[768, 3072]", permute_136: "f32[3072, 768]", div_7: "f32[8, 197, 1]", permute_140: "f32[768, 768]", alias_15: "f32[8, 12, 197, 64]", permute_146: "f32[2304, 768]", div_8: "f32[8, 197, 1]", permute_150: "f32[768, 3072]", permute_154: "f32[3072, 768]", div_9: "f32[8, 197, 1]", permute_158: "f32[768, 768]", alias_16: "f32[8, 12, 197, 64]", permute_164: "f32[2304, 768]", div_10: "f32[8, 197, 1]", permute_168: "f32[768, 3072]", permute_172: "f32[3072, 768]", div_11: "f32[8, 197, 1]", permute_176: "f32[768, 768]", alias_17: "f32[8, 12, 197, 64]", permute_182: "f32[2304, 768]", div_12: "f32[8, 197, 1]", permute_186: "f32[768, 3072]", permute_190: "f32[3072, 768]", div_13: "f32[8, 197, 1]", permute_194: "f32[768, 768]", alias_18: "f32[8, 12, 197, 64]", permute_200: "f32[2304, 768]", div_14: "f32[8, 197, 1]", permute_204: "f32[768, 3072]", permute_208: "f32[3072, 768]", div_15: "f32[8, 197, 1]", permute_212: "f32[768, 768]", alias_19: "f32[8, 12, 197, 64]", permute_218: "f32[2304, 768]", div_16: "f32[8, 197, 1]", permute_222: "f32[768, 3072]", permute_226: "f32[3072, 768]", div_17: "f32[8, 197, 1]", permute_230: "f32[768, 768]", alias_20: "f32[8, 12, 197, 64]", permute_236: "f32[2304, 768]", div_18: "f32[8, 197, 1]", permute_240: "f32[768, 3072]", permute_244: "f32[3072, 768]", div_19: "f32[8, 197, 1]", permute_248: "f32[768, 768]", alias_21: "f32[8, 12, 197, 64]", permute_254: "f32[2304, 768]", div_20: "f32[8, 197, 1]", permute_258: "f32[768, 3072]", permute_262: "f32[3072, 768]", div_21: "f32[8, 197, 1]", permute_266: "f32[768, 768]", alias_22: "f32[8, 12, 197, 64]", permute_272: "f32[2304, 768]", div_22: "f32[8, 197, 1]", permute_276: "f32[768, 3072]", permute_280: "f32[3072, 768]", div_23: "f32[8, 197, 1]", permute_284: "f32[768, 768]", alias_23: "f32[8, 12, 197, 64]", permute_290: "f32[2304, 768]", div_24: "f32[8, 197, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_8: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_12: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_28: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_48: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_68: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_54: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_55: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_61: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_98: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_108: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_76: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_83: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:649, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_74);  permute_74 = None
    permute_75: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_75, clone_37);  permute_75 = clone_37 = None
    permute_76: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_121: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_77: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:646, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_default: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 197, 768]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 0);  mm = None
    slice_scatter: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, select_scatter, 0, 0, 9223372036854775807);  full_default = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_149);  primals_149 = None
    mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_87, 768)
    sum_2: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_87, mul_84);  mul_87 = None
    sum_3: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
    mul_90: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_3);  sum_3 = None
    sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_88, sum_2);  mul_88 = sum_2 = None
    sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
    mul_91: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_84);  mul_84 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[1576, 768]" = torch.ops.aten.view.default(mul_91, [1576, 768])
    mm_2: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_122, permute_78);  permute_78 = None
    permute_79: "f32[768, 1576]" = torch.ops.aten.permute.default(view_122, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_79, view_119);  permute_79 = view_119 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
    view_123: "f32[768]" = torch.ops.aten.view.default(sum_6, [768]);  sum_6 = None
    permute_81: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_2, [8, 197, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_94: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_83, 0.5);  add_83 = None
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
    exp: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_98: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_97);  view_118 = mul_97 = None
    add_88: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
    mul_99: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, add_88);  view_124 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_99, [1576, 3072]);  mul_99 = None
    mm_4: "f32[1576, 768]" = torch.ops.aten.mm.default(view_125, permute_82);  permute_82 = None
    permute_83: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_125, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_83, view_117);  permute_83 = view_117 = None
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
    view_126: "f32[3072]" = torch.ops.aten.view.default(sum_7, [3072]);  sum_7 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_4, [8, 197, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, primals_143);  primals_143 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
    sum_8: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_79);  mul_101 = None
    sum_9: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_9);  sum_9 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_8);  mul_102 = sum_8 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
    mul_105: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_30);  div_1 = sub_30 = None
    mul_106: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_79);  mul_79 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_128: "f32[1576, 768]" = torch.ops.aten.view.default(add_89, [1576, 768])
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_128, permute_86);  permute_86 = None
    permute_87: "f32[768, 1576]" = torch.ops.aten.permute.default(view_128, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_87, view_115);  permute_87 = view_115 = None
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True);  view_128 = None
    view_129: "f32[768]" = torch.ops.aten.view.default(sum_12, [768]);  sum_12 = None
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    view_130: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_6, [8, 197, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_131: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_130, [8, 197, 12, 64]);  view_130 = None
    permute_90: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_90, getitem_123, getitem_124, getitem_125, None, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False]);  permute_90 = getitem_123 = getitem_124 = getitem_125 = alias_12 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_134: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_135: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_136: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_134, getitem_135, getitem_136]);  getitem_134 = getitem_135 = getitem_136 = None
    view_132: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_1, [3, 8, 12, 197, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_91: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_132, [1, 3, 0, 2, 4]);  view_132 = None
    clone_38: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_133: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_38, [8, 197, 2304]);  clone_38 = None
    view_134: "f32[1576, 2304]" = torch.ops.aten.view.default(view_133, [1576, 2304]);  view_133 = None
    mm_8: "f32[1576, 768]" = torch.ops.aten.mm.default(view_134, permute_92);  permute_92 = None
    permute_93: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_134, [1, 0])
    mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_93, view_111);  permute_93 = view_111 = None
    permute_94: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
    view_135: "f32[2304]" = torch.ops.aten.view.default(sum_13, [2304]);  sum_13 = None
    permute_95: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    view_136: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_8, [8, 197, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_136, primals_137);  primals_137 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_14: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_77);  mul_108 = None
    sum_15: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_15);  sum_15 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_14);  mul_109 = sum_14 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_33);  div_2 = sub_33 = None
    mul_113: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_136, mul_77);  mul_77 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_136, [0, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_90: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_89, mul_112);  add_89 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[1576, 768]" = torch.ops.aten.view.default(add_90, [1576, 768])
    mm_10: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_137, permute_96);  permute_96 = None
    permute_97: "f32[768, 1576]" = torch.ops.aten.permute.default(view_137, [1, 0])
    mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_97, view_109);  permute_97 = view_109 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
    view_138: "f32[768]" = torch.ops.aten.view.default(sum_18, [768]);  sum_18 = None
    permute_99: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    view_139: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_10, [8, 197, 3072]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_76, 0.5);  add_76 = None
    mul_116: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, view_108)
    mul_117: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
    exp_1: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, mul_118);  view_108 = mul_118 = None
    add_92: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
    mul_120: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_139, add_92);  view_139 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_120, [1576, 3072]);  mul_120 = None
    mm_12: "f32[1576, 768]" = torch.ops.aten.mm.default(view_140, permute_100);  permute_100 = None
    permute_101: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_140, [1, 0])
    mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_101, view_107);  permute_101 = view_107 = None
    permute_102: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
    view_141: "f32[3072]" = torch.ops.aten.view.default(sum_19, [3072]);  sum_19 = None
    permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_142: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_12, [8, 197, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_142, primals_131);  primals_131 = None
    mul_123: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_20: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_72);  mul_122 = None
    sum_21: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_21);  sum_21 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_20);  mul_123 = sum_20 = None
    sub_36: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
    mul_126: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_142, mul_72);  mul_72 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_90, mul_126);  add_90 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_143: "f32[1576, 768]" = torch.ops.aten.view.default(add_93, [1576, 768])
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_143, permute_104);  permute_104 = None
    permute_105: "f32[768, 1576]" = torch.ops.aten.permute.default(view_143, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_105, view_105);  permute_105 = view_105 = None
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
    view_144: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_145: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_14, [8, 197, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_146: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_145, [8, 197, 12, 64]);  view_145 = None
    permute_108: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_108, getitem_112, getitem_113, getitem_114, None, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False]);  permute_108 = getitem_112 = getitem_113 = getitem_114 = alias_13 = getitem_116 = getitem_117 = getitem_118 = None
    getitem_138: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_139: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_140: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140]);  getitem_138 = getitem_139 = getitem_140 = None
    view_147: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_2, [3, 8, 12, 197, 64]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_109: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_147, [1, 3, 0, 2, 4]);  view_147 = None
    clone_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_148: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_39, [8, 197, 2304]);  clone_39 = None
    view_149: "f32[1576, 2304]" = torch.ops.aten.view.default(view_148, [1576, 2304]);  view_148 = None
    mm_16: "f32[1576, 768]" = torch.ops.aten.mm.default(view_149, permute_110);  permute_110 = None
    permute_111: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_149, [1, 0])
    mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_111, view_101);  permute_111 = view_101 = None
    permute_112: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
    view_150: "f32[2304]" = torch.ops.aten.view.default(sum_25, [2304]);  sum_25 = None
    permute_113: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_151: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_16, [8, 197, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_129: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_151, primals_125);  primals_125 = None
    mul_130: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_129, 768)
    sum_26: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_70);  mul_129 = None
    sum_27: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_27);  sum_27 = None
    sub_38: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_130, sum_26);  mul_130 = sum_26 = None
    sub_39: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
    mul_133: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
    mul_134: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_151, mul_70);  mul_70 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_151, [0, 1]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_94: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_93, mul_133);  add_93 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1576, 768]" = torch.ops.aten.view.default(add_94, [1576, 768])
    mm_18: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_152, permute_114);  permute_114 = None
    permute_115: "f32[768, 1576]" = torch.ops.aten.permute.default(view_152, [1, 0])
    mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_115, view_99);  permute_115 = view_99 = None
    permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
    view_153: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    permute_117: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    view_154: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_18, [8, 197, 3072]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_136: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_137: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_138: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
    exp_2: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
    mul_139: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_140: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_139);  view_98 = mul_139 = None
    add_96: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
    mul_141: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_154, add_96);  view_154 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_155: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_141, [1576, 3072]);  mul_141 = None
    mm_20: "f32[1576, 768]" = torch.ops.aten.mm.default(view_155, permute_118);  permute_118 = None
    permute_119: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_155, [1, 0])
    mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_119, view_97);  permute_119 = view_97 = None
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_155, [0], True);  view_155 = None
    view_156: "f32[3072]" = torch.ops.aten.view.default(sum_31, [3072]);  sum_31 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    view_157: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_20, [8, 197, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_157, primals_119);  primals_119 = None
    mul_144: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_32: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_65);  mul_143 = None
    sum_33: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_33);  sum_33 = None
    sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_32);  mul_144 = sum_32 = None
    sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_157, mul_65);  mul_65 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_157, [0, 1]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_97: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_94, mul_147);  add_94 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_158: "f32[1576, 768]" = torch.ops.aten.view.default(add_97, [1576, 768])
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_158, permute_122);  permute_122 = None
    permute_123: "f32[768, 1576]" = torch.ops.aten.permute.default(view_158, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_123, view_95);  permute_123 = view_95 = None
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True);  view_158 = None
    view_159: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_160: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_161: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_160, [8, 197, 12, 64]);  view_160 = None
    permute_126: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_126, getitem_101, getitem_102, getitem_103, None, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False]);  permute_126 = getitem_101 = getitem_102 = getitem_103 = alias_14 = getitem_105 = getitem_106 = getitem_107 = None
    getitem_142: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_143: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_144: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_142, getitem_143, getitem_144]);  getitem_142 = getitem_143 = getitem_144 = None
    view_162: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_3, [3, 8, 12, 197, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_127: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_162, [1, 3, 0, 2, 4]);  view_162 = None
    clone_40: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_163: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_40, [8, 197, 2304]);  clone_40 = None
    view_164: "f32[1576, 2304]" = torch.ops.aten.view.default(view_163, [1576, 2304]);  view_163 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_164, permute_128);  permute_128 = None
    permute_129: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_164, [1, 0])
    mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_129, view_91);  permute_129 = view_91 = None
    permute_130: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_37: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
    view_165: "f32[2304]" = torch.ops.aten.view.default(sum_37, [2304]);  sum_37 = None
    permute_131: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_166: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_150: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_166, primals_113);  primals_113 = None
    mul_151: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_38: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_63);  mul_150 = None
    sum_39: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_39);  sum_39 = None
    sub_44: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_38);  mul_151 = sum_38 = None
    sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_45);  div_6 = sub_45 = None
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_166, mul_63);  mul_63 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_166, [0, 1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_98: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_97, mul_154);  add_97 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[1576, 768]" = torch.ops.aten.view.default(add_98, [1576, 768])
    mm_26: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_167, permute_132);  permute_132 = None
    permute_133: "f32[768, 1576]" = torch.ops.aten.permute.default(view_167, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_133, view_89);  permute_133 = view_89 = None
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
    view_168: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_135: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_169: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_26, [8, 197, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_157: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_158: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_159: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
    exp_3: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
    mul_160: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_161: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_160);  view_88 = mul_160 = None
    add_100: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
    mul_162: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_169, add_100);  view_169 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_170: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_162, [1576, 3072]);  mul_162 = None
    mm_28: "f32[1576, 768]" = torch.ops.aten.mm.default(view_170, permute_136);  permute_136 = None
    permute_137: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_170, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_137, view_87);  permute_137 = view_87 = None
    permute_138: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
    view_171: "f32[3072]" = torch.ops.aten.view.default(sum_43, [3072]);  sum_43 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_172: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_28, [8, 197, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_172, primals_107);  primals_107 = None
    mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_44: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_58);  mul_164 = None
    sum_45: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_45);  sum_45 = None
    sub_47: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_44);  mul_165 = sum_44 = None
    sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_48);  div_7 = sub_48 = None
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_172, mul_58);  mul_58 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_172, [0, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_98, mul_168);  add_98 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_173: "f32[1576, 768]" = torch.ops.aten.view.default(add_101, [1576, 768])
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_173, permute_140);  permute_140 = None
    permute_141: "f32[768, 1576]" = torch.ops.aten.permute.default(view_173, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_141, view_85);  permute_141 = view_85 = None
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
    view_174: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    view_175: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_30, [8, 197, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_176: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_175, [8, 197, 12, 64]);  view_175 = None
    permute_144: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_144, getitem_90, getitem_91, getitem_92, None, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False]);  permute_144 = getitem_90 = getitem_91 = getitem_92 = alias_15 = getitem_94 = getitem_95 = getitem_96 = None
    getitem_146: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_147: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_148: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
    view_177: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_4, [3, 8, 12, 197, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_145: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_177, [1, 3, 0, 2, 4]);  view_177 = None
    clone_41: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_178: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_41, [8, 197, 2304]);  clone_41 = None
    view_179: "f32[1576, 2304]" = torch.ops.aten.view.default(view_178, [1576, 2304]);  view_178 = None
    mm_32: "f32[1576, 768]" = torch.ops.aten.mm.default(view_179, permute_146);  permute_146 = None
    permute_147: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_179, [1, 0])
    mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_147, view_81);  permute_147 = view_81 = None
    permute_148: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
    view_180: "f32[2304]" = torch.ops.aten.view.default(sum_49, [2304]);  sum_49 = None
    permute_149: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_181: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_32, [8, 197, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, primals_101);  primals_101 = None
    mul_172: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
    sum_50: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_56);  mul_171 = None
    sum_51: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_56, sum_51);  sum_51 = None
    sub_50: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_50);  mul_172 = sum_50 = None
    sub_51: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
    mul_175: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_51);  div_8 = sub_51 = None
    mul_176: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, mul_56);  mul_56 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_181, [0, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, mul_175);  add_101 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1576, 768]" = torch.ops.aten.view.default(add_102, [1576, 768])
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_182, permute_150);  permute_150 = None
    permute_151: "f32[768, 1576]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_79);  permute_151 = view_79 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_184: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_178: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_55, 0.5);  add_55 = None
    mul_179: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_180: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
    exp_4: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
    mul_181: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_182: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_181);  view_78 = mul_181 = None
    add_104: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
    mul_183: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_184, add_104);  view_184 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_183, [1576, 3072]);  mul_183 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_185, permute_154);  permute_154 = None
    permute_155: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_185, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_77);  permute_155 = view_77 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[3072]" = torch.ops.aten.view.default(sum_55, [3072]);  sum_55 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_187: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_185: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_187, primals_95);  primals_95 = None
    mul_186: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
    sum_56: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
    mul_187: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_51);  mul_185 = None
    sum_57: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
    mul_188: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_57);  sum_57 = None
    sub_53: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_56);  mul_186 = sum_56 = None
    sub_54: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
    mul_189: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_54);  div_9 = sub_54 = None
    mul_190: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_187, mul_51);  mul_51 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_102, mul_189);  add_102 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_188: "f32[1576, 768]" = torch.ops.aten.view.default(add_105, [1576, 768])
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_188, permute_158);  permute_158 = None
    permute_159: "f32[768, 1576]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_159, view_75);  permute_159 = view_75 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_190: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_191: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_190, [8, 197, 12, 64]);  view_190 = None
    permute_162: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_162, getitem_79, getitem_80, getitem_81, None, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_162 = getitem_79 = getitem_80 = getitem_81 = alias_16 = getitem_83 = getitem_84 = getitem_85 = None
    getitem_150: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_151: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_152: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
    view_192: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_5, [3, 8, 12, 197, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_163: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_192, [1, 3, 0, 2, 4]);  view_192 = None
    clone_42: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_193: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_42, [8, 197, 2304]);  clone_42 = None
    view_194: "f32[1576, 2304]" = torch.ops.aten.view.default(view_193, [1576, 2304]);  view_193 = None
    mm_40: "f32[1576, 768]" = torch.ops.aten.mm.default(view_194, permute_164);  permute_164 = None
    permute_165: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_165, view_71);  permute_165 = view_71 = None
    permute_166: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_61: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[2304]" = torch.ops.aten.view.default(sum_61, [2304]);  sum_61 = None
    permute_167: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    view_196: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_40, [8, 197, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_192: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_196, primals_89);  primals_89 = None
    mul_193: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_49);  mul_192 = None
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_63);  sum_63 = None
    sub_56: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_62);  mul_193 = sum_62 = None
    sub_57: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
    mul_196: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_57);  div_10 = sub_57 = None
    mul_197: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_196, mul_49);  mul_49 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_196, [0, 1]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_106: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_105, mul_196);  add_105 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_197: "f32[1576, 768]" = torch.ops.aten.view.default(add_106, [1576, 768])
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_197, permute_168);  permute_168 = None
    permute_169: "f32[768, 1576]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_169, view_69);  permute_169 = view_69 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_171: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_199: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_199: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_200: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_201: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
    exp_5: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
    mul_202: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_203: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, mul_202);  view_68 = mul_202 = None
    add_108: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
    mul_204: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_199, add_108);  view_199 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_200: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_204, [1576, 3072]);  mul_204 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_200, permute_172);  permute_172 = None
    permute_173: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_173, view_67);  permute_173 = view_67 = None
    permute_174: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[3072]" = torch.ops.aten.view.default(sum_67, [3072]);  sum_67 = None
    permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_202: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_206: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_202, primals_83);  primals_83 = None
    mul_207: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
    sum_68: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_44);  mul_206 = None
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_69);  sum_69 = None
    sub_59: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_68);  mul_207 = sum_68 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_60);  div_11 = sub_60 = None
    mul_211: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_202, mul_44);  mul_44 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_202, [0, 1]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_106, mul_210);  add_106 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_203: "f32[1576, 768]" = torch.ops.aten.view.default(add_109, [1576, 768])
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_203, permute_176);  permute_176 = None
    permute_177: "f32[768, 1576]" = torch.ops.aten.permute.default(view_203, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_177, view_65);  permute_177 = view_65 = None
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
    view_204: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_179: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_205: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_206: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_205, [8, 197, 12, 64]);  view_205 = None
    permute_180: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_180, getitem_68, getitem_69, getitem_70, None, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False]);  permute_180 = getitem_68 = getitem_69 = getitem_70 = alias_17 = getitem_72 = getitem_73 = getitem_74 = None
    getitem_154: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_155: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_156: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
    view_207: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_6, [3, 8, 12, 197, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_181: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_207, [1, 3, 0, 2, 4]);  view_207 = None
    clone_43: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_208: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_43, [8, 197, 2304]);  clone_43 = None
    view_209: "f32[1576, 2304]" = torch.ops.aten.view.default(view_208, [1576, 2304]);  view_208 = None
    mm_48: "f32[1576, 768]" = torch.ops.aten.mm.default(view_209, permute_182);  permute_182 = None
    permute_183: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_183, view_61);  permute_183 = view_61 = None
    permute_184: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[2304]" = torch.ops.aten.view.default(sum_73, [2304]);  sum_73 = None
    permute_185: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_48, [8, 197, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_213: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, primals_77);  primals_77 = None
    mul_214: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_74: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_42);  mul_213 = None
    sum_75: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_42, sum_75);  sum_75 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_74);  mul_214 = sum_74 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
    mul_218: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_42);  mul_42 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_109, mul_217);  add_109 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1576, 768]" = torch.ops.aten.view.default(add_110, [1576, 768])
    mm_50: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_212, permute_186);  permute_186 = None
    permute_187: "f32[768, 1576]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_187, view_59);  permute_187 = view_59 = None
    permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    permute_189: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_50, [8, 197, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_220: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
    mul_221: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_222: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_6: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_224: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_223);  view_58 = mul_223 = None
    add_112: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, add_112);  view_214 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_215: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_225, [1576, 3072]);  mul_225 = None
    mm_52: "f32[1576, 768]" = torch.ops.aten.mm.default(view_215, permute_190);  permute_190 = None
    permute_191: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_191, view_57);  permute_191 = view_57 = None
    permute_192: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[3072]" = torch.ops.aten.view.default(sum_79, [3072]);  sum_79 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_217: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_52, [8, 197, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_227: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, primals_71);  primals_71 = None
    mul_228: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
    sum_80: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_37);  mul_227 = None
    sum_81: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_81);  sum_81 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_80);  mul_228 = sum_80 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
    mul_231: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_66);  div_13 = sub_66 = None
    mul_232: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_37);  mul_37 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, mul_231);  add_110 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_218: "f32[1576, 768]" = torch.ops.aten.view.default(add_113, [1576, 768])
    mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_218, permute_194);  permute_194 = None
    permute_195: "f32[768, 1576]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_55);  permute_195 = view_55 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[768]" = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_220: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_54, [8, 197, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_221: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_220, [8, 197, 12, 64]);  view_220 = None
    permute_198: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_198, getitem_57, getitem_58, getitem_59, None, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_198 = getitem_57 = getitem_58 = getitem_59 = alias_18 = getitem_61 = getitem_62 = getitem_63 = None
    getitem_158: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_159: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_160: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
    view_222: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_7, [3, 8, 12, 197, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_199: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_222, [1, 3, 0, 2, 4]);  view_222 = None
    clone_44: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_223: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_44, [8, 197, 2304]);  clone_44 = None
    view_224: "f32[1576, 2304]" = torch.ops.aten.view.default(view_223, [1576, 2304]);  view_223 = None
    mm_56: "f32[1576, 768]" = torch.ops.aten.mm.default(view_224, permute_200);  permute_200 = None
    permute_201: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_201, view_51);  permute_201 = view_51 = None
    permute_202: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_85: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[2304]" = torch.ops.aten.view.default(sum_85, [2304]);  sum_85 = None
    permute_203: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_226: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_56, [8, 197, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_234: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_226, primals_65);  primals_65 = None
    mul_235: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_234, 768)
    sum_86: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_234, mul_35);  mul_234 = None
    sum_87: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_87);  sum_87 = None
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_235, sum_86);  mul_235 = sum_86 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
    mul_238: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_69);  div_14 = sub_69 = None
    mul_239: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_226, mul_35);  mul_35 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_113, mul_238);  add_113 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_227: "f32[1576, 768]" = torch.ops.aten.view.default(add_114, [1576, 768])
    mm_58: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_227, permute_204);  permute_204 = None
    permute_205: "f32[768, 1576]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_205, view_49);  permute_205 = view_49 = None
    permute_206: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_229: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_58, [8, 197, 3072]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_241: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_242: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, view_48)
    mul_243: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_7: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_245: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, mul_244);  view_48 = mul_244 = None
    add_116: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_229, add_116);  view_229 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_230: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_246, [1576, 3072]);  mul_246 = None
    mm_60: "f32[1576, 768]" = torch.ops.aten.mm.default(view_230, permute_208);  permute_208 = None
    permute_209: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_230, [1, 0])
    mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_209, view_47);  permute_209 = view_47 = None
    permute_210: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[3072]" = torch.ops.aten.view.default(sum_91, [3072]);  sum_91 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_232: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_60, [8, 197, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_248: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_232, primals_59);  primals_59 = None
    mul_249: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_92: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_30);  mul_248 = None
    sum_93: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_93);  sum_93 = None
    sub_71: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_92);  mul_249 = sum_92 = None
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
    mul_252: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_72);  div_15 = sub_72 = None
    mul_253: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_232, mul_30);  mul_30 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_232, [0, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_114, mul_252);  add_114 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_233: "f32[1576, 768]" = torch.ops.aten.view.default(add_117, [1576, 768])
    mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_233, permute_212);  permute_212 = None
    permute_213: "f32[768, 1576]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_213, view_45);  permute_213 = view_45 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_215: "f32[768, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_235: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_62, [8, 197, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_236: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_235, [8, 197, 12, 64]);  view_235 = None
    permute_216: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_216, getitem_46, getitem_47, getitem_48, None, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False]);  permute_216 = getitem_46 = getitem_47 = getitem_48 = alias_19 = getitem_50 = getitem_51 = getitem_52 = None
    getitem_162: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_163: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_164: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
    view_237: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_8, [3, 8, 12, 197, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_217: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_237, [1, 3, 0, 2, 4]);  view_237 = None
    clone_45: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    view_238: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_45, [8, 197, 2304]);  clone_45 = None
    view_239: "f32[1576, 2304]" = torch.ops.aten.view.default(view_238, [1576, 2304]);  view_238 = None
    mm_64: "f32[1576, 768]" = torch.ops.aten.mm.default(view_239, permute_218);  permute_218 = None
    permute_219: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_219, view_41);  permute_219 = view_41 = None
    permute_220: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_97: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[2304]" = torch.ops.aten.view.default(sum_97, [2304]);  sum_97 = None
    permute_221: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_241: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_64, [8, 197, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_255: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, primals_53);  primals_53 = None
    mul_256: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_98: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_28);  mul_255 = None
    sum_99: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_28, sum_99);  sum_99 = None
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_98);  mul_256 = sum_98 = None
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
    mul_259: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_75);  div_16 = sub_75 = None
    mul_260: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, mul_28);  mul_28 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_118: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_117, mul_259);  add_117 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_242: "f32[1576, 768]" = torch.ops.aten.view.default(add_118, [1576, 768])
    mm_66: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_242, permute_222);  permute_222 = None
    permute_223: "f32[768, 1576]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_223, view_39);  permute_223 = view_39 = None
    permute_224: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_244: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_66, [8, 197, 3072]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_262: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_27, 0.5);  add_27 = None
    mul_263: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_264: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_8: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_266: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_265);  view_38 = mul_265 = None
    add_120: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_244, add_120);  view_244 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_245: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_267, [1576, 3072]);  mul_267 = None
    mm_68: "f32[1576, 768]" = torch.ops.aten.mm.default(view_245, permute_226);  permute_226 = None
    permute_227: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_245, [1, 0])
    mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_227, view_37);  permute_227 = view_37 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
    view_246: "f32[3072]" = torch.ops.aten.view.default(sum_103, [3072]);  sum_103 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_247: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_68, [8, 197, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_269: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_247, primals_47);  primals_47 = None
    mul_270: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_104: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_23);  mul_269 = None
    sum_105: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_105);  sum_105 = None
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_104);  mul_270 = sum_104 = None
    sub_78: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
    mul_273: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
    mul_274: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_247, mul_23);  mul_23 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_247, [0, 1]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_121: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_118, mul_273);  add_118 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_248: "f32[1576, 768]" = torch.ops.aten.view.default(add_121, [1576, 768])
    mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_248, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_35);  permute_231 = view_35 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_250: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_70, [8, 197, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_251: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_250, [8, 197, 12, 64]);  view_250 = None
    permute_234: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_234, getitem_35, getitem_36, getitem_37, None, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False]);  permute_234 = getitem_35 = getitem_36 = getitem_37 = alias_20 = getitem_39 = getitem_40 = getitem_41 = None
    getitem_166: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_167: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_168: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
    view_252: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_9, [3, 8, 12, 197, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_235: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_252, [1, 3, 0, 2, 4]);  view_252 = None
    clone_46: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_253: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_46, [8, 197, 2304]);  clone_46 = None
    view_254: "f32[1576, 2304]" = torch.ops.aten.view.default(view_253, [1576, 2304]);  view_253 = None
    mm_72: "f32[1576, 768]" = torch.ops.aten.mm.default(view_254, permute_236);  permute_236 = None
    permute_237: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_237, view_31);  permute_237 = view_31 = None
    permute_238: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_109: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[2304]" = torch.ops.aten.view.default(sum_109, [2304]);  sum_109 = None
    permute_239: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_256: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_72, [8, 197, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_276: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_256, primals_41);  primals_41 = None
    mul_277: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
    sum_110: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
    mul_278: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_21);  mul_276 = None
    sum_111: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
    mul_279: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_111);  sum_111 = None
    sub_80: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_110);  mul_277 = sum_110 = None
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
    mul_280: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
    mul_281: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_256, mul_21);  mul_21 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_121, mul_280);  add_121 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_257: "f32[1576, 768]" = torch.ops.aten.view.default(add_122, [1576, 768])
    mm_74: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_257, permute_240);  permute_240 = None
    permute_241: "f32[768, 1576]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_29);  permute_241 = view_29 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    permute_243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_259: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_74, [8, 197, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_283: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_284: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, view_28)
    mul_285: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
    exp_9: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
    mul_286: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_287: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, mul_286);  view_28 = mul_286 = None
    add_124: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
    mul_288: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_259, add_124);  view_259 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_260: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_288, [1576, 3072]);  mul_288 = None
    mm_76: "f32[1576, 768]" = torch.ops.aten.mm.default(view_260, permute_244);  permute_244 = None
    permute_245: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_27);  permute_245 = view_27 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[3072]" = torch.ops.aten.view.default(sum_115, [3072]);  sum_115 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_262: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_76, [8, 197, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_290: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_262, primals_35);  primals_35 = None
    mul_291: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_290, 768)
    sum_116: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
    mul_292: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_16);  mul_290 = None
    sum_117: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
    mul_293: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_117);  sum_117 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_291, sum_116);  mul_291 = sum_116 = None
    sub_84: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
    mul_294: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_84);  div_19 = sub_84 = None
    mul_295: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_262, mul_16);  mul_16 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_262, [0, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_122, mul_294);  add_122 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_263: "f32[1576, 768]" = torch.ops.aten.view.default(add_125, [1576, 768])
    mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_263, permute_248);  permute_248 = None
    permute_249: "f32[768, 1576]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_249, view_25);  permute_249 = view_25 = None
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_265: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_78, [8, 197, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_266: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_265, [8, 197, 12, 64]);  view_265 = None
    permute_252: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_252, getitem_24, getitem_25, getitem_26, None, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False]);  permute_252 = getitem_24 = getitem_25 = getitem_26 = alias_21 = getitem_28 = getitem_29 = getitem_30 = None
    getitem_170: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
    getitem_171: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
    getitem_172: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
    view_267: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_10, [3, 8, 12, 197, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_253: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_267, [1, 3, 0, 2, 4]);  view_267 = None
    clone_47: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    view_268: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_47, [8, 197, 2304]);  clone_47 = None
    view_269: "f32[1576, 2304]" = torch.ops.aten.view.default(view_268, [1576, 2304]);  view_268 = None
    mm_80: "f32[1576, 768]" = torch.ops.aten.mm.default(view_269, permute_254);  permute_254 = None
    permute_255: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_255, view_21);  permute_255 = view_21 = None
    permute_256: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_121: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[2304]" = torch.ops.aten.view.default(sum_121, [2304]);  sum_121 = None
    permute_257: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_271: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_80, [8, 197, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_297: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, primals_29);  primals_29 = None
    mul_298: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_122: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_14);  mul_297 = None
    sum_123: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_123);  sum_123 = None
    sub_86: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_122);  mul_298 = sum_122 = None
    sub_87: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
    mul_301: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_87);  div_20 = sub_87 = None
    mul_302: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_14);  mul_14 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_126: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, mul_301);  add_125 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_272: "f32[1576, 768]" = torch.ops.aten.view.default(add_126, [1576, 768])
    mm_82: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_272, permute_258);  permute_258 = None
    permute_259: "f32[768, 1576]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_259, view_19);  permute_259 = view_19 = None
    permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_82, [8, 197, 3072]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_304: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_305: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_306: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_10: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_308: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_307);  view_18 = mul_307 = None
    add_128: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, add_128);  view_274 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_309, [1576, 3072]);  mul_309 = None
    mm_84: "f32[1576, 768]" = torch.ops.aten.mm.default(view_275, permute_262);  permute_262 = None
    permute_263: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_263, view_17);  permute_263 = view_17 = None
    permute_264: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_127: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[3072]" = torch.ops.aten.view.default(sum_127, [3072]);  sum_127 = None
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_277: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_84, [8, 197, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_311: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_277, primals_23);  primals_23 = None
    mul_312: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_128: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_9);  mul_311 = None
    sum_129: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_129);  sum_129 = None
    sub_89: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_128);  mul_312 = sum_128 = None
    sub_90: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
    mul_315: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
    mul_316: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_277, mul_9);  mul_9 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_277, [0, 1]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_126, mul_315);  add_126 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_278: "f32[1576, 768]" = torch.ops.aten.view.default(add_129, [1576, 768])
    mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_278, permute_266);  permute_266 = None
    permute_267: "f32[768, 1576]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_15);  permute_267 = view_15 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_280: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_86, [8, 197, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_281: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_280, [8, 197, 12, 64]);  view_280 = None
    permute_270: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_270, getitem_13, getitem_14, getitem_15, None, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False]);  permute_270 = getitem_13 = getitem_14 = getitem_15 = alias_22 = getitem_17 = getitem_18 = getitem_19 = None
    getitem_174: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
    getitem_175: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
    getitem_176: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
    view_282: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_11, [3, 8, 12, 197, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_271: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_282, [1, 3, 0, 2, 4]);  view_282 = None
    clone_48: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
    view_283: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_48, [8, 197, 2304]);  clone_48 = None
    view_284: "f32[1576, 2304]" = torch.ops.aten.view.default(view_283, [1576, 2304]);  view_283 = None
    mm_88: "f32[1576, 768]" = torch.ops.aten.mm.default(view_284, permute_272);  permute_272 = None
    permute_273: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_273, view_11);  permute_273 = view_11 = None
    permute_274: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_133: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[2304]" = torch.ops.aten.view.default(sum_133, [2304]);  sum_133 = None
    permute_275: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_88, [8, 197, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_318: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_286, primals_17);  primals_17 = None
    mul_319: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
    sum_134: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_318, mul_7);  mul_318 = None
    sum_135: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_7, sum_135);  sum_135 = None
    sub_92: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_134);  mul_319 = sum_134 = None
    sub_93: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
    mul_322: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_93);  div_22 = sub_93 = None
    mul_323: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_286, mul_7);  mul_7 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_286, [0, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_130: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_322);  add_129 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.view.default(add_130, [1576, 768])
    mm_90: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_287, permute_276);  permute_276 = None
    permute_277: "f32[768, 1576]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_277, view_9);  permute_277 = view_9 = None
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_279: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_289: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_90, [8, 197, 3072]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_325: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_326: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_327: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_11: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_329: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, mul_328);  view_8 = mul_328 = None
    add_132: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_289, add_132);  view_289 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_290: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_330, [1576, 3072]);  mul_330 = None
    mm_92: "f32[1576, 768]" = torch.ops.aten.mm.default(view_290, permute_280);  permute_280 = None
    permute_281: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_281, view_7);  permute_281 = view_7 = None
    permute_282: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[3072]" = torch.ops.aten.view.default(sum_139, [3072]);  sum_139 = None
    permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_292: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_92, [8, 197, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_332: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_292, primals_11);  primals_11 = None
    mul_333: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
    sum_140: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_2);  mul_332 = None
    sum_141: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_141);  sum_141 = None
    sub_95: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_140);  mul_333 = sum_140 = None
    sub_96: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
    mul_336: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_96);  div_23 = sub_96 = None
    mul_337: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_292, mul_2);  mul_2 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_292, [0, 1]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_133: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_130, mul_336);  add_130 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_293: "f32[1576, 768]" = torch.ops.aten.view.default(add_133, [1576, 768])
    mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_293, permute_284);  permute_284 = None
    permute_285: "f32[768, 1576]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_285, view_5);  permute_285 = view_5 = None
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[768]" = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
    permute_287: "f32[768, 768]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_295: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_94, [8, 197, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_296: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_295, [8, 197, 12, 64]);  view_295 = None
    permute_288: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_288, getitem_2, getitem_3, getitem_4, None, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False]);  permute_288 = getitem_2 = getitem_3 = getitem_4 = alias_23 = getitem_6 = getitem_7 = getitem_8 = None
    getitem_178: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
    getitem_179: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
    getitem_180: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
    view_297: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_12, [3, 8, 12, 197, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_297, [1, 3, 0, 2, 4]);  view_297 = None
    clone_49: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_298: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_49, [8, 197, 2304]);  clone_49 = None
    view_299: "f32[1576, 2304]" = torch.ops.aten.view.default(view_298, [1576, 2304]);  view_298 = None
    mm_96: "f32[1576, 768]" = torch.ops.aten.mm.default(view_299, permute_290);  permute_290 = None
    permute_291: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_291, view_1);  permute_291 = view_1 = None
    permute_292: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_145: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[2304]" = torch.ops.aten.view.default(sum_145, [2304]);  sum_145 = None
    permute_293: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_301: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_96, [8, 197, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_301, primals_5);  primals_5 = None
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_339, 768)
    sum_146: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul);  mul_339 = None
    sum_147: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, sum_147);  sum_147 = None
    sub_98: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_340, sum_146);  mul_340 = sum_146 = None
    sub_99: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_99);  div_24 = sub_99 = None
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_301, mul);  mul = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_134: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_133, mul_343);  add_133 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:580, code: x = x + pos_embed
    sum_150: "f32[1, 197, 768]" = torch.ops.aten.sum.dim_IntList(add_134, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:579, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    slice_2: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_134, 1, 0, 1)
    slice_3: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_134, 1, 1, 197);  add_134 = None
    sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_2, [0], True);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_294: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_3, [0, 2, 1]);  slice_3 = None
    view_302: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_294, [8, 768, 14, 14]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_302, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_302, primals_153, primals_3, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_302 = primals_153 = primals_3 = None
    getitem_183: "f32[768, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return [sum_150, sum_151, getitem_183, sum_152, sum_148, sum_149, permute_293, view_300, permute_287, view_294, sum_142, sum_143, permute_283, view_291, permute_279, view_288, sum_136, sum_137, permute_275, view_285, permute_269, view_279, sum_130, sum_131, permute_265, view_276, permute_261, view_273, sum_124, sum_125, permute_257, view_270, permute_251, view_264, sum_118, sum_119, permute_247, view_261, permute_243, view_258, sum_112, sum_113, permute_239, view_255, permute_233, view_249, sum_106, sum_107, permute_229, view_246, permute_225, view_243, sum_100, sum_101, permute_221, view_240, permute_215, view_234, sum_94, sum_95, permute_211, view_231, permute_207, view_228, sum_88, sum_89, permute_203, view_225, permute_197, view_219, sum_82, sum_83, permute_193, view_216, permute_189, view_213, sum_76, sum_77, permute_185, view_210, permute_179, view_204, sum_70, sum_71, permute_175, view_201, permute_171, view_198, sum_64, sum_65, permute_167, view_195, permute_161, view_189, sum_58, sum_59, permute_157, view_186, permute_153, view_183, sum_52, sum_53, permute_149, view_180, permute_143, view_174, sum_46, sum_47, permute_139, view_171, permute_135, view_168, sum_40, sum_41, permute_131, view_165, permute_125, view_159, sum_34, sum_35, permute_121, view_156, permute_117, view_153, sum_28, sum_29, permute_113, view_150, permute_107, view_144, sum_22, sum_23, permute_103, view_141, permute_99, view_138, sum_16, sum_17, permute_95, view_135, permute_89, view_129, sum_10, sum_11, permute_85, view_126, permute_81, view_123, sum_4, sum_5, permute_77, view_121, None]
    