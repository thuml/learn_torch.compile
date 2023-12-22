from __future__ import annotations



def forward(self, primals_1: "f32[768, 3, 16, 16]", primals_3: "f32[768]", primals_6: "f32[384]", primals_9: "f32[768]", primals_15: "f32[768]", primals_18: "f32[384]", primals_21: "f32[768]", primals_27: "f32[768]", primals_30: "f32[384]", primals_33: "f32[768]", primals_39: "f32[768]", primals_42: "f32[384]", primals_45: "f32[768]", primals_51: "f32[768]", primals_54: "f32[384]", primals_57: "f32[768]", primals_63: "f32[768]", primals_66: "f32[384]", primals_69: "f32[768]", primals_75: "f32[768]", primals_78: "f32[384]", primals_81: "f32[768]", primals_87: "f32[768]", primals_90: "f32[384]", primals_93: "f32[768]", primals_99: "f32[768]", primals_102: "f32[384]", primals_105: "f32[768]", primals_111: "f32[768]", primals_114: "f32[384]", primals_117: "f32[768]", primals_123: "f32[768]", primals_126: "f32[384]", primals_129: "f32[768]", primals_135: "f32[768]", primals_138: "f32[384]", primals_141: "f32[768]", primals_147: "f32[768]", primals_151: "f32[8, 3, 224, 224]", mul: "f32[8, 196, 768]", view_1: "f32[6144, 196]", mm: "f32[6144, 384]", view_3: "f32[6144, 384]", mul_5: "f32[8, 196, 768]", view_5: "f32[1568, 768]", addmm_1: "f32[1568, 3072]", view_7: "f32[1568, 3072]", mul_10: "f32[8, 196, 768]", view_9: "f32[6144, 196]", mm_1: "f32[6144, 384]", view_11: "f32[6144, 384]", mul_15: "f32[8, 196, 768]", view_13: "f32[1568, 768]", addmm_4: "f32[1568, 3072]", view_15: "f32[1568, 3072]", mul_20: "f32[8, 196, 768]", view_17: "f32[6144, 196]", mm_2: "f32[6144, 384]", view_19: "f32[6144, 384]", mul_25: "f32[8, 196, 768]", view_21: "f32[1568, 768]", addmm_7: "f32[1568, 3072]", view_23: "f32[1568, 3072]", mul_30: "f32[8, 196, 768]", view_25: "f32[6144, 196]", mm_3: "f32[6144, 384]", view_27: "f32[6144, 384]", mul_35: "f32[8, 196, 768]", view_29: "f32[1568, 768]", addmm_10: "f32[1568, 3072]", view_31: "f32[1568, 3072]", mul_40: "f32[8, 196, 768]", view_33: "f32[6144, 196]", mm_4: "f32[6144, 384]", view_35: "f32[6144, 384]", mul_45: "f32[8, 196, 768]", view_37: "f32[1568, 768]", addmm_13: "f32[1568, 3072]", view_39: "f32[1568, 3072]", mul_50: "f32[8, 196, 768]", view_41: "f32[6144, 196]", mm_5: "f32[6144, 384]", view_43: "f32[6144, 384]", mul_55: "f32[8, 196, 768]", view_45: "f32[1568, 768]", addmm_16: "f32[1568, 3072]", view_47: "f32[1568, 3072]", mul_60: "f32[8, 196, 768]", view_49: "f32[6144, 196]", mm_6: "f32[6144, 384]", view_51: "f32[6144, 384]", mul_65: "f32[8, 196, 768]", view_53: "f32[1568, 768]", addmm_19: "f32[1568, 3072]", view_55: "f32[1568, 3072]", mul_70: "f32[8, 196, 768]", view_57: "f32[6144, 196]", mm_7: "f32[6144, 384]", view_59: "f32[6144, 384]", mul_75: "f32[8, 196, 768]", view_61: "f32[1568, 768]", addmm_22: "f32[1568, 3072]", view_63: "f32[1568, 3072]", mul_80: "f32[8, 196, 768]", view_65: "f32[6144, 196]", mm_8: "f32[6144, 384]", view_67: "f32[6144, 384]", mul_85: "f32[8, 196, 768]", view_69: "f32[1568, 768]", addmm_25: "f32[1568, 3072]", view_71: "f32[1568, 3072]", mul_90: "f32[8, 196, 768]", view_73: "f32[6144, 196]", mm_9: "f32[6144, 384]", view_75: "f32[6144, 384]", mul_95: "f32[8, 196, 768]", view_77: "f32[1568, 768]", addmm_28: "f32[1568, 3072]", view_79: "f32[1568, 3072]", mul_100: "f32[8, 196, 768]", view_81: "f32[6144, 196]", mm_10: "f32[6144, 384]", view_83: "f32[6144, 384]", mul_105: "f32[8, 196, 768]", view_85: "f32[1568, 768]", addmm_31: "f32[1568, 3072]", view_87: "f32[1568, 3072]", mul_110: "f32[8, 196, 768]", view_89: "f32[6144, 196]", mm_11: "f32[6144, 384]", view_91: "f32[6144, 384]", mul_115: "f32[8, 196, 768]", view_93: "f32[1568, 768]", addmm_34: "f32[1568, 3072]", view_95: "f32[1568, 3072]", mul_120: "f32[8, 196, 768]", clone_85: "f32[8, 768]", permute_74: "f32[1000, 768]", div_1: "f32[8, 196, 1]", permute_78: "f32[768, 3072]", permute_82: "f32[3072, 768]", div_2: "f32[8, 196, 1]", permute_87: "f32[196, 384]", permute_93: "f32[384, 196]", div_3: "f32[8, 196, 1]", permute_96: "f32[768, 3072]", permute_100: "f32[3072, 768]", div_4: "f32[8, 196, 1]", permute_105: "f32[196, 384]", permute_111: "f32[384, 196]", div_5: "f32[8, 196, 1]", permute_114: "f32[768, 3072]", permute_118: "f32[3072, 768]", div_6: "f32[8, 196, 1]", permute_123: "f32[196, 384]", permute_129: "f32[384, 196]", div_7: "f32[8, 196, 1]", permute_132: "f32[768, 3072]", permute_136: "f32[3072, 768]", div_8: "f32[8, 196, 1]", permute_141: "f32[196, 384]", permute_147: "f32[384, 196]", div_9: "f32[8, 196, 1]", permute_150: "f32[768, 3072]", permute_154: "f32[3072, 768]", div_10: "f32[8, 196, 1]", permute_159: "f32[196, 384]", permute_165: "f32[384, 196]", div_11: "f32[8, 196, 1]", permute_168: "f32[768, 3072]", permute_172: "f32[3072, 768]", div_12: "f32[8, 196, 1]", permute_177: "f32[196, 384]", permute_183: "f32[384, 196]", div_13: "f32[8, 196, 1]", permute_186: "f32[768, 3072]", permute_190: "f32[3072, 768]", div_14: "f32[8, 196, 1]", permute_195: "f32[196, 384]", permute_201: "f32[384, 196]", div_15: "f32[8, 196, 1]", permute_204: "f32[768, 3072]", permute_208: "f32[3072, 768]", div_16: "f32[8, 196, 1]", permute_213: "f32[196, 384]", permute_219: "f32[384, 196]", div_17: "f32[8, 196, 1]", permute_222: "f32[768, 3072]", permute_226: "f32[3072, 768]", div_18: "f32[8, 196, 1]", permute_231: "f32[196, 384]", permute_237: "f32[384, 196]", div_19: "f32[8, 196, 1]", permute_240: "f32[768, 3072]", permute_244: "f32[3072, 768]", div_20: "f32[8, 196, 1]", permute_249: "f32[196, 384]", permute_255: "f32[384, 196]", div_21: "f32[8, 196, 1]", permute_258: "f32[768, 3072]", permute_262: "f32[3072, 768]", div_22: "f32[8, 196, 1]", permute_267: "f32[196, 384]", permute_273: "f32[384, 196]", div_23: "f32[8, 196, 1]", permute_276: "f32[768, 3072]", permute_280: "f32[3072, 768]", div_24: "f32[8, 196, 1]", permute_285: "f32[196, 384]", permute_291: "f32[384, 196]", div_25: "f32[8, 196, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_2: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm, [8, 768, 384]);  mm = None
    add_2: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_2, primals_6);  view_2 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_3: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, 0.7071067811865476)
    erf: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_3: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_6: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 196, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, 0.7071067811865476)
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_7: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_1, [8, 768, 384]);  mm_1 = None
    add_11: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_10, primals_18);  view_10 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_13: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, 0.7071067811865476)
    erf_2: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_12: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_14: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 196, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, 0.7071067811865476)
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_16: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_2, [8, 768, 384]);  mm_2 = None
    add_20: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_18, primals_30);  view_18 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, 0.7071067811865476)
    erf_4: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_21: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 196, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476)
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_25: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_26: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 768, 384]);  mm_3 = None
    add_29: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_26, primals_42);  view_26 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_33: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, 0.7071067811865476)
    erf_6: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_30: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_30: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 196, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, 0.7071067811865476)
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_34: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_34: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_4, [8, 768, 384]);  mm_4 = None
    add_38: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_34, primals_54);  view_34 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_43: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, 0.7071067811865476)
    erf_8: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_39: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 196, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_43: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_5, [8, 768, 384]);  mm_5 = None
    add_47: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_42, primals_66);  view_42 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, 0.7071067811865476)
    erf_10: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_48: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 196, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_11: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_52: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_6, [8, 768, 384]);  mm_6 = None
    add_56: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_50, primals_78);  view_50 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_63: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, 0.7071067811865476)
    erf_12: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_57: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_54: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 196, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, 0.7071067811865476)
    erf_13: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_61: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_7, [8, 768, 384]);  mm_7 = None
    add_65: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_58, primals_90);  view_58 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, 0.7071067811865476)
    erf_14: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_73);  mul_73 = None
    add_66: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 196, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476)
    erf_15: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_70: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_66: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_8, [8, 768, 384]);  mm_8 = None
    add_74: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_66, primals_102);  view_66 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_83: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, 0.7071067811865476)
    erf_16: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_75: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 196, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476)
    erf_17: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_74: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_9, [8, 768, 384]);  mm_9 = None
    add_83: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_74, primals_114);  view_74 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, 0.7071067811865476)
    erf_18: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_84: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_78: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 196, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_19: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_88: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_82: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_10, [8, 768, 384]);  mm_10 = None
    add_92: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_82, primals_126);  view_82 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_103: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, 0.7071067811865476)
    erf_20: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_93: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_86: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 196, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_108: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, 0.7071067811865476)
    erf_21: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_97: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_90: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_11, [8, 768, 384]);  mm_11 = None
    add_101: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(view_90, primals_138);  view_90 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_113: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, 0.7071067811865476)
    erf_22: "f32[8, 768, 384]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_102: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_94: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 196, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_118: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476)
    erf_23: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_106: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:273, code: return x if pre_logits else self.head(x)
    mm_12: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_74);  permute_74 = None
    permute_75: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_13: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_75, clone_85);  permute_75 = clone_85 = None
    permute_76: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_97: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_77: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:271, code: x = x.mean(dim=1)
    unsqueeze: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(mm_12, 1);  mm_12 = None
    expand: "f32[8, 196, 768]" = torch.ops.aten.expand.default(unsqueeze, [8, 196, 768]);  unsqueeze = None
    div: "f32[8, 196, 768]" = torch.ops.aten.div.Scalar(expand, 196);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:266, code: x = self.norm(x)
    mul_123: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div, primals_147);  primals_147 = None
    mul_124: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_123, 768)
    sum_2: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
    mul_125: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_123, mul_120);  mul_123 = None
    sum_3: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
    mul_126: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_120, sum_3);  sum_3 = None
    sub_26: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_124, sum_2);  mul_124 = sum_2 = None
    sub_27: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_126);  sub_26 = mul_126 = None
    mul_127: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_27);  div_1 = sub_27 = None
    mul_128: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div, mul_120);  mul_120 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(div, [0, 1]);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_98: "f32[1568, 768]" = torch.ops.aten.reshape.default(mul_127, [1568, 768])
    mm_14: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_98, permute_78);  permute_78 = None
    permute_79: "f32[768, 1568]" = torch.ops.aten.permute.default(view_98, [1, 0])
    mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_79, view_95);  permute_79 = view_95 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_98, [0], True);  view_98 = None
    view_99: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
    permute_81: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    view_100: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_14, [8, 196, 3072]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_130: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_131: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, view_94)
    mul_132: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_134: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_94, mul_133);  view_94 = mul_133 = None
    add_111: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_100, add_111);  view_100 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_101: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_135, [1568, 3072]);  mul_135 = None
    mm_16: "f32[1568, 768]" = torch.ops.aten.mm.default(view_101, permute_82);  permute_82 = None
    permute_83: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_101, [1, 0])
    mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_83, view_93);  permute_83 = view_93 = None
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_101, [0], True);  view_101 = None
    view_102: "f32[3072]" = torch.ops.aten.reshape.default(sum_7, [3072]);  sum_7 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    view_103: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_16, [8, 196, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_137: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_103, primals_141);  primals_141 = None
    mul_138: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
    sum_8: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_115);  mul_137 = None
    sum_9: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_115, sum_9);  sum_9 = None
    sub_29: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_8);  mul_138 = sum_8 = None
    sub_30: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_140);  sub_29 = mul_140 = None
    mul_141: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_30);  div_2 = sub_30 = None
    mul_142: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_103, mul_115);  mul_115 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_103, [0, 1]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_112: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(mul_127, mul_141);  mul_127 = mul_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_86: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_112, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_88: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
    view_104: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_88, [6144, 196]);  clone_88 = None
    mm_18: "f32[6144, 384]" = torch.ops.aten.mm.default(view_104, permute_87);  permute_87 = None
    permute_88: "f32[196, 6144]" = torch.ops.aten.permute.default(view_104, [1, 0])
    mm_19: "f32[196, 384]" = torch.ops.aten.mm.default(permute_88, view_91);  permute_88 = view_91 = None
    permute_89: "f32[384, 196]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_12: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_104, [0], True);  view_104 = None
    view_105: "f32[196]" = torch.ops.aten.reshape.default(sum_12, [196]);  sum_12 = None
    permute_90: "f32[196, 384]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    view_106: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_18, [8, 768, 384]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_144: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_145: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, add_101)
    mul_146: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_145, -0.5);  mul_145 = None
    exp_1: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_146);  mul_146 = None
    mul_147: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_148: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_101, mul_147);  add_101 = mul_147 = None
    add_114: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_144, mul_148);  mul_144 = mul_148 = None
    mul_149: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_106, add_114);  view_106 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_13: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 1], True)
    view_107: "f32[384]" = torch.ops.aten.reshape.default(sum_13, [384]);  sum_13 = None
    view_108: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_149, [6144, 384]);  mul_149 = None
    permute_91: "f32[384, 6144]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_20: "f32[384, 196]" = torch.ops.aten.mm.default(permute_91, view_89);  permute_91 = view_89 = None
    permute_92: "f32[196, 384]" = torch.ops.aten.permute.default(mm_20, [1, 0]);  mm_20 = None
    mm_21: "f32[6144, 196]" = torch.ops.aten.mm.default(view_108, permute_93);  view_108 = permute_93 = None
    view_109: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_21, [8, 768, 196]);  mm_21 = None
    permute_94: "f32[384, 196]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_95: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_109, [0, 2, 1]);  view_109 = None
    clone_89: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    mul_151: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_89, primals_135);  primals_135 = None
    mul_152: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_151, 768)
    sum_14: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True)
    mul_153: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_110);  mul_151 = None
    sum_15: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
    mul_154: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_15);  sum_15 = None
    sub_32: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_152, sum_14);  mul_152 = sum_14 = None
    sub_33: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_154);  sub_32 = mul_154 = None
    mul_155: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_33);  div_3 = sub_33 = None
    mul_156: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_89, mul_110);  mul_110 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1]);  mul_156 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_89, [0, 1]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_115: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_112, mul_155);  add_112 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_110: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_115, [1568, 768])
    mm_22: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_110, permute_96);  permute_96 = None
    permute_97: "f32[768, 1568]" = torch.ops.aten.permute.default(view_110, [1, 0])
    mm_23: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_97, view_87);  permute_97 = view_87 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_110, [0], True);  view_110 = None
    view_111: "f32[768]" = torch.ops.aten.reshape.default(sum_18, [768]);  sum_18 = None
    permute_99: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    view_112: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_22, [8, 196, 3072]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_158: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_159: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, view_86)
    mul_160: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_159, -0.5);  mul_159 = None
    exp_2: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_160);  mul_160 = None
    mul_161: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_162: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_86, mul_161);  view_86 = mul_161 = None
    add_117: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_158, mul_162);  mul_158 = mul_162 = None
    mul_163: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_112, add_117);  view_112 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_113: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_163, [1568, 3072]);  mul_163 = None
    mm_24: "f32[1568, 768]" = torch.ops.aten.mm.default(view_113, permute_100);  permute_100 = None
    permute_101: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_113, [1, 0])
    mm_25: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_101, view_85);  permute_101 = view_85 = None
    permute_102: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_113, [0], True);  view_113 = None
    view_114: "f32[3072]" = torch.ops.aten.reshape.default(sum_19, [3072]);  sum_19 = None
    permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_115: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_24, [8, 196, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_165: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_115, primals_129);  primals_129 = None
    mul_166: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, 768)
    sum_20: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True)
    mul_167: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_165, mul_105);  mul_165 = None
    sum_21: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True);  mul_167 = None
    mul_168: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_105, sum_21);  sum_21 = None
    sub_35: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_166, sum_20);  mul_166 = sum_20 = None
    sub_36: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_168);  sub_35 = mul_168 = None
    mul_169: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_36);  div_4 = sub_36 = None
    mul_170: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_115, mul_105);  mul_105 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_170, [0, 1]);  mul_170 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_115, [0, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_118: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_115, mul_169);  add_115 = mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_104: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_118, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_92: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_116: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_92, [6144, 196]);  clone_92 = None
    mm_26: "f32[6144, 384]" = torch.ops.aten.mm.default(view_116, permute_105);  permute_105 = None
    permute_106: "f32[196, 6144]" = torch.ops.aten.permute.default(view_116, [1, 0])
    mm_27: "f32[196, 384]" = torch.ops.aten.mm.default(permute_106, view_83);  permute_106 = view_83 = None
    permute_107: "f32[384, 196]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_24: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True);  view_116 = None
    view_117: "f32[196]" = torch.ops.aten.reshape.default(sum_24, [196]);  sum_24 = None
    permute_108: "f32[196, 384]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_118: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_26, [8, 768, 384]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_172: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_93, 0.5);  add_93 = None
    mul_173: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, add_92)
    mul_174: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_173, -0.5);  mul_173 = None
    exp_3: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_174);  mul_174 = None
    mul_175: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_176: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_92, mul_175);  add_92 = mul_175 = None
    add_120: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_172, mul_176);  mul_172 = mul_176 = None
    mul_177: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_118, add_120);  view_118 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_25: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 1], True)
    view_119: "f32[384]" = torch.ops.aten.reshape.default(sum_25, [384]);  sum_25 = None
    view_120: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_177, [6144, 384]);  mul_177 = None
    permute_109: "f32[384, 6144]" = torch.ops.aten.permute.default(view_120, [1, 0])
    mm_28: "f32[384, 196]" = torch.ops.aten.mm.default(permute_109, view_81);  permute_109 = view_81 = None
    permute_110: "f32[196, 384]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    mm_29: "f32[6144, 196]" = torch.ops.aten.mm.default(view_120, permute_111);  view_120 = permute_111 = None
    view_121: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_29, [8, 768, 196]);  mm_29 = None
    permute_112: "f32[384, 196]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_113: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_121, [0, 2, 1]);  view_121 = None
    clone_93: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    mul_179: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_93, primals_123);  primals_123 = None
    mul_180: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_26: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_100);  mul_179 = None
    sum_27: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_27);  sum_27 = None
    sub_38: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_26);  mul_180 = sum_26 = None
    sub_39: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_182);  sub_38 = mul_182 = None
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_39);  div_5 = sub_39 = None
    mul_184: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_93, mul_100);  mul_100 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_93, [0, 1]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_121: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_118, mul_183);  add_118 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_121, [1568, 768])
    mm_30: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_122, permute_114);  permute_114 = None
    permute_115: "f32[768, 1568]" = torch.ops.aten.permute.default(view_122, [1, 0])
    mm_31: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_115, view_79);  permute_115 = view_79 = None
    permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
    view_123: "f32[768]" = torch.ops.aten.reshape.default(sum_30, [768]);  sum_30 = None
    permute_117: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    view_124: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_30, [8, 196, 3072]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_186: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_187: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_188: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_187, -0.5);  mul_187 = None
    exp_4: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_188);  mul_188 = None
    mul_189: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_190: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_189);  view_78 = mul_189 = None
    add_123: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_186, mul_190);  mul_186 = mul_190 = None
    mul_191: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_124, add_123);  view_124 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_191, [1568, 3072]);  mul_191 = None
    mm_32: "f32[1568, 768]" = torch.ops.aten.mm.default(view_125, permute_118);  permute_118 = None
    permute_119: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_125, [1, 0])
    mm_33: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_119, view_77);  permute_119 = view_77 = None
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
    view_126: "f32[3072]" = torch.ops.aten.reshape.default(sum_31, [3072]);  sum_31 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    view_127: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_32, [8, 196, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_193: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_127, primals_117);  primals_117 = None
    mul_194: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_193, 768)
    sum_32: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
    mul_195: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_193, mul_95);  mul_193 = None
    sum_33: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    mul_196: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, sum_33);  sum_33 = None
    sub_41: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_194, sum_32);  mul_194 = sum_32 = None
    sub_42: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_196);  sub_41 = mul_196 = None
    mul_197: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_42);  div_6 = sub_42 = None
    mul_198: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_95);  mul_95 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_124: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_121, mul_197);  add_121 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_122: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_124, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_96: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_128: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_96, [6144, 196]);  clone_96 = None
    mm_34: "f32[6144, 384]" = torch.ops.aten.mm.default(view_128, permute_123);  permute_123 = None
    permute_124: "f32[196, 6144]" = torch.ops.aten.permute.default(view_128, [1, 0])
    mm_35: "f32[196, 384]" = torch.ops.aten.mm.default(permute_124, view_75);  permute_124 = view_75 = None
    permute_125: "f32[384, 196]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_36: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True);  view_128 = None
    view_129: "f32[196]" = torch.ops.aten.reshape.default(sum_36, [196]);  sum_36 = None
    permute_126: "f32[196, 384]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_130: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_34, [8, 768, 384]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_200: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_201: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, add_83)
    mul_202: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_201, -0.5);  mul_201 = None
    exp_5: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_202);  mul_202 = None
    mul_203: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_204: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_83, mul_203);  add_83 = mul_203 = None
    add_126: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_200, mul_204);  mul_200 = mul_204 = None
    mul_205: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_130, add_126);  view_130 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_37: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1], True)
    view_131: "f32[384]" = torch.ops.aten.reshape.default(sum_37, [384]);  sum_37 = None
    view_132: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_205, [6144, 384]);  mul_205 = None
    permute_127: "f32[384, 6144]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_36: "f32[384, 196]" = torch.ops.aten.mm.default(permute_127, view_73);  permute_127 = view_73 = None
    permute_128: "f32[196, 384]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    mm_37: "f32[6144, 196]" = torch.ops.aten.mm.default(view_132, permute_129);  view_132 = permute_129 = None
    view_133: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_37, [8, 768, 196]);  mm_37 = None
    permute_130: "f32[384, 196]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_131: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    clone_97: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_97, primals_111);  primals_111 = None
    mul_208: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_207, 768)
    sum_38: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True)
    mul_209: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_207, mul_90);  mul_207 = None
    sum_39: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True);  mul_209 = None
    mul_210: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_39);  sum_39 = None
    sub_44: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_208, sum_38);  mul_208 = sum_38 = None
    sub_45: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_210);  sub_44 = mul_210 = None
    mul_211: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_45);  div_7 = sub_45 = None
    mul_212: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_97, mul_90);  mul_90 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 1]);  mul_212 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_97, [0, 1]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_127: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_124, mul_211);  add_124 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_134: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_127, [1568, 768])
    mm_38: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_134, permute_132);  permute_132 = None
    permute_133: "f32[768, 1568]" = torch.ops.aten.permute.default(view_134, [1, 0])
    mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_133, view_71);  permute_133 = view_71 = None
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
    view_135: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
    permute_135: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_136: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_38, [8, 196, 3072]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_214: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.5);  add_79 = None
    mul_215: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, view_70)
    mul_216: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_215, -0.5);  mul_215 = None
    exp_6: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_216);  mul_216 = None
    mul_217: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_218: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_70, mul_217);  view_70 = mul_217 = None
    add_129: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_214, mul_218);  mul_214 = mul_218 = None
    mul_219: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_136, add_129);  view_136 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_137: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_219, [1568, 3072]);  mul_219 = None
    mm_40: "f32[1568, 768]" = torch.ops.aten.mm.default(view_137, permute_136);  permute_136 = None
    permute_137: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_137, [1, 0])
    mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_137, view_69);  permute_137 = view_69 = None
    permute_138: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
    view_138: "f32[3072]" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_139: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_40, [8, 196, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_139, primals_105);  primals_105 = None
    mul_222: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, 768)
    sum_44: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True)
    mul_223: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_221, mul_85);  mul_221 = None
    sum_45: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_223, [2], True);  mul_223 = None
    mul_224: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_45);  sum_45 = None
    sub_47: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_222, sum_44);  mul_222 = sum_44 = None
    sub_48: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_224);  sub_47 = mul_224 = None
    mul_225: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_48);  div_8 = sub_48 = None
    mul_226: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_139, mul_85);  mul_85 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_226, [0, 1]);  mul_226 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_139, [0, 1]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_130: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_127, mul_225);  add_127 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_140: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_130, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_100: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_140: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_100, [6144, 196]);  clone_100 = None
    mm_42: "f32[6144, 384]" = torch.ops.aten.mm.default(view_140, permute_141);  permute_141 = None
    permute_142: "f32[196, 6144]" = torch.ops.aten.permute.default(view_140, [1, 0])
    mm_43: "f32[196, 384]" = torch.ops.aten.mm.default(permute_142, view_67);  permute_142 = view_67 = None
    permute_143: "f32[384, 196]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_48: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
    view_141: "f32[196]" = torch.ops.aten.reshape.default(sum_48, [196]);  sum_48 = None
    permute_144: "f32[196, 384]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_142: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_42, [8, 768, 384]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_228: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_229: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, add_74)
    mul_230: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_229, -0.5);  mul_229 = None
    exp_7: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_230);  mul_230 = None
    mul_231: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_232: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_74, mul_231);  add_74 = mul_231 = None
    add_132: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_228, mul_232);  mul_228 = mul_232 = None
    mul_233: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_142, add_132);  view_142 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_49: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_233, [0, 1], True)
    view_143: "f32[384]" = torch.ops.aten.reshape.default(sum_49, [384]);  sum_49 = None
    view_144: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_233, [6144, 384]);  mul_233 = None
    permute_145: "f32[384, 6144]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_44: "f32[384, 196]" = torch.ops.aten.mm.default(permute_145, view_65);  permute_145 = view_65 = None
    permute_146: "f32[196, 384]" = torch.ops.aten.permute.default(mm_44, [1, 0]);  mm_44 = None
    mm_45: "f32[6144, 196]" = torch.ops.aten.mm.default(view_144, permute_147);  view_144 = permute_147 = None
    view_145: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_45, [8, 768, 196]);  mm_45 = None
    permute_148: "f32[384, 196]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_149: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    clone_101: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
    mul_235: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_101, primals_99);  primals_99 = None
    mul_236: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_235, 768)
    sum_50: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [2], True)
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_235, mul_80);  mul_235 = None
    sum_51: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True);  mul_237 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_51);  sum_51 = None
    sub_50: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_236, sum_50);  mul_236 = sum_50 = None
    sub_51: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_238);  sub_50 = mul_238 = None
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_51);  div_9 = sub_51 = None
    mul_240: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_101, mul_80);  mul_80 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1]);  mul_240 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_101, [0, 1]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_133: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_130, mul_239);  add_130 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_146: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_133, [1568, 768])
    mm_46: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_146, permute_150);  permute_150 = None
    permute_151: "f32[768, 1568]" = torch.ops.aten.permute.default(view_146, [1, 0])
    mm_47: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_63);  permute_151 = view_63 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_146, [0], True);  view_146 = None
    view_147: "f32[768]" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_148: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_46, [8, 196, 3072]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_242: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
    mul_243: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, view_62)
    mul_244: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_243, -0.5);  mul_243 = None
    exp_8: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_244);  mul_244 = None
    mul_245: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_246: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_62, mul_245);  view_62 = mul_245 = None
    add_135: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_242, mul_246);  mul_242 = mul_246 = None
    mul_247: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_148, add_135);  view_148 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_149: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_247, [1568, 3072]);  mul_247 = None
    mm_48: "f32[1568, 768]" = torch.ops.aten.mm.default(view_149, permute_154);  permute_154 = None
    permute_155: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_149, [1, 0])
    mm_49: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_61);  permute_155 = view_61 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
    view_150: "f32[3072]" = torch.ops.aten.reshape.default(sum_55, [3072]);  sum_55 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_151: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 196, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_249: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_151, primals_93);  primals_93 = None
    mul_250: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_249, 768)
    sum_56: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True)
    mul_251: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_249, mul_75);  mul_249 = None
    sum_57: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, sum_57);  sum_57 = None
    sub_53: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_250, sum_56);  mul_250 = sum_56 = None
    sub_54: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_252);  sub_53 = mul_252 = None
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_54);  div_10 = sub_54 = None
    mul_254: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_151, mul_75);  mul_75 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1]);  mul_254 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_151, [0, 1]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_136: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_133, mul_253);  add_133 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_158: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_136, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_104: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_152: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_104, [6144, 196]);  clone_104 = None
    mm_50: "f32[6144, 384]" = torch.ops.aten.mm.default(view_152, permute_159);  permute_159 = None
    permute_160: "f32[196, 6144]" = torch.ops.aten.permute.default(view_152, [1, 0])
    mm_51: "f32[196, 384]" = torch.ops.aten.mm.default(permute_160, view_59);  permute_160 = view_59 = None
    permute_161: "f32[384, 196]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_60: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
    view_153: "f32[196]" = torch.ops.aten.reshape.default(sum_60, [196]);  sum_60 = None
    permute_162: "f32[196, 384]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_154: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_50, [8, 768, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_256: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
    mul_257: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, add_65)
    mul_258: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_257, -0.5);  mul_257 = None
    exp_9: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_258);  mul_258 = None
    mul_259: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_260: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_65, mul_259);  add_65 = mul_259 = None
    add_138: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_256, mul_260);  mul_256 = mul_260 = None
    mul_261: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_154, add_138);  view_154 = add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_61: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1], True)
    view_155: "f32[384]" = torch.ops.aten.reshape.default(sum_61, [384]);  sum_61 = None
    view_156: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_261, [6144, 384]);  mul_261 = None
    permute_163: "f32[384, 6144]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_52: "f32[384, 196]" = torch.ops.aten.mm.default(permute_163, view_57);  permute_163 = view_57 = None
    permute_164: "f32[196, 384]" = torch.ops.aten.permute.default(mm_52, [1, 0]);  mm_52 = None
    mm_53: "f32[6144, 196]" = torch.ops.aten.mm.default(view_156, permute_165);  view_156 = permute_165 = None
    view_157: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_53, [8, 768, 196]);  mm_53 = None
    permute_166: "f32[384, 196]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_167: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
    clone_105: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    mul_263: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_105, primals_87);  primals_87 = None
    mul_264: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_263, 768)
    sum_62: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True)
    mul_265: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_263, mul_70);  mul_263 = None
    sum_63: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [2], True);  mul_265 = None
    mul_266: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_63);  sum_63 = None
    sub_56: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_264, sum_62);  mul_264 = sum_62 = None
    sub_57: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_266);  sub_56 = mul_266 = None
    mul_267: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_57);  div_11 = sub_57 = None
    mul_268: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_105, mul_70);  mul_70 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_105, [0, 1]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_139: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_136, mul_267);  add_136 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_158: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_139, [1568, 768])
    mm_54: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_158, permute_168);  permute_168 = None
    permute_169: "f32[768, 1568]" = torch.ops.aten.permute.default(view_158, [1, 0])
    mm_55: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_169, view_55);  permute_169 = view_55 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True);  view_158 = None
    view_159: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    permute_171: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_160: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_54, [8, 196, 3072]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_270: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
    mul_271: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, view_54)
    mul_272: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_271, -0.5);  mul_271 = None
    exp_10: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_272);  mul_272 = None
    mul_273: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_274: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_54, mul_273);  view_54 = mul_273 = None
    add_141: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_270, mul_274);  mul_270 = mul_274 = None
    mul_275: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_160, add_141);  view_160 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_161: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_275, [1568, 3072]);  mul_275 = None
    mm_56: "f32[1568, 768]" = torch.ops.aten.mm.default(view_161, permute_172);  permute_172 = None
    permute_173: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_161, [1, 0])
    mm_57: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_173, view_53);  permute_173 = view_53 = None
    permute_174: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
    view_162: "f32[3072]" = torch.ops.aten.reshape.default(sum_67, [3072]);  sum_67 = None
    permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_163: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_56, [8, 196, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_163, primals_81);  primals_81 = None
    mul_278: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_277, 768)
    sum_68: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
    mul_279: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_277, mul_65);  mul_277 = None
    sum_69: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    mul_280: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_69);  sum_69 = None
    sub_59: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_278, sum_68);  mul_278 = sum_68 = None
    sub_60: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_280);  sub_59 = mul_280 = None
    mul_281: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_60);  div_12 = sub_60 = None
    mul_282: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_163, mul_65);  mul_65 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_163, [0, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_142: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_139, mul_281);  add_139 = mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_176: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_142, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_108: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
    view_164: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_108, [6144, 196]);  clone_108 = None
    mm_58: "f32[6144, 384]" = torch.ops.aten.mm.default(view_164, permute_177);  permute_177 = None
    permute_178: "f32[196, 6144]" = torch.ops.aten.permute.default(view_164, [1, 0])
    mm_59: "f32[196, 384]" = torch.ops.aten.mm.default(permute_178, view_51);  permute_178 = view_51 = None
    permute_179: "f32[384, 196]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_72: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
    view_165: "f32[196]" = torch.ops.aten.reshape.default(sum_72, [196]);  sum_72 = None
    permute_180: "f32[196, 384]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    view_166: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_58, [8, 768, 384]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_284: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_57, 0.5);  add_57 = None
    mul_285: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, add_56)
    mul_286: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_285, -0.5);  mul_285 = None
    exp_11: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_286);  mul_286 = None
    mul_287: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_288: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_56, mul_287);  add_56 = mul_287 = None
    add_144: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_284, mul_288);  mul_284 = mul_288 = None
    mul_289: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_166, add_144);  view_166 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_73: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1], True)
    view_167: "f32[384]" = torch.ops.aten.reshape.default(sum_73, [384]);  sum_73 = None
    view_168: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_289, [6144, 384]);  mul_289 = None
    permute_181: "f32[384, 6144]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_60: "f32[384, 196]" = torch.ops.aten.mm.default(permute_181, view_49);  permute_181 = view_49 = None
    permute_182: "f32[196, 384]" = torch.ops.aten.permute.default(mm_60, [1, 0]);  mm_60 = None
    mm_61: "f32[6144, 196]" = torch.ops.aten.mm.default(view_168, permute_183);  view_168 = permute_183 = None
    view_169: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_61, [8, 768, 196]);  mm_61 = None
    permute_184: "f32[384, 196]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_185: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    clone_109: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
    mul_291: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_109, primals_75);  primals_75 = None
    mul_292: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_74: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_60);  mul_291 = None
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_75);  sum_75 = None
    sub_62: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_74);  mul_292 = sum_74 = None
    sub_63: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_294);  sub_62 = mul_294 = None
    mul_295: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_63);  div_13 = sub_63 = None
    mul_296: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_109, mul_60);  mul_60 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_109, [0, 1]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_145: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_142, mul_295);  add_142 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_170: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_145, [1568, 768])
    mm_62: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_170, permute_186);  permute_186 = None
    permute_187: "f32[768, 1568]" = torch.ops.aten.permute.default(view_170, [1, 0])
    mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_187, view_47);  permute_187 = view_47 = None
    permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
    view_171: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
    permute_189: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    view_172: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_62, [8, 196, 3072]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_298: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_52, 0.5);  add_52 = None
    mul_299: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_300: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_299, -0.5);  mul_299 = None
    exp_12: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_300);  mul_300 = None
    mul_301: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_302: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_46, mul_301);  view_46 = mul_301 = None
    add_147: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_298, mul_302);  mul_298 = mul_302 = None
    mul_303: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_172, add_147);  view_172 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_173: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_303, [1568, 3072]);  mul_303 = None
    mm_64: "f32[1568, 768]" = torch.ops.aten.mm.default(view_173, permute_190);  permute_190 = None
    permute_191: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_173, [1, 0])
    mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_191, view_45);  permute_191 = view_45 = None
    permute_192: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_79: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
    view_174: "f32[3072]" = torch.ops.aten.reshape.default(sum_79, [3072]);  sum_79 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_175: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_64, [8, 196, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_305: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_175, primals_69);  primals_69 = None
    mul_306: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_305, 768)
    sum_80: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
    mul_307: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_55);  mul_305 = None
    sum_81: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
    mul_308: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_81);  sum_81 = None
    sub_65: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_306, sum_80);  mul_306 = sum_80 = None
    sub_66: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_308);  sub_65 = mul_308 = None
    mul_309: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_66);  div_14 = sub_66 = None
    mul_310: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_175, mul_55);  mul_55 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_175, [0, 1]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_148: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_145, mul_309);  add_145 = mul_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_194: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_148, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_112: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_176: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_112, [6144, 196]);  clone_112 = None
    mm_66: "f32[6144, 384]" = torch.ops.aten.mm.default(view_176, permute_195);  permute_195 = None
    permute_196: "f32[196, 6144]" = torch.ops.aten.permute.default(view_176, [1, 0])
    mm_67: "f32[196, 384]" = torch.ops.aten.mm.default(permute_196, view_43);  permute_196 = view_43 = None
    permute_197: "f32[384, 196]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_84: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True);  view_176 = None
    view_177: "f32[196]" = torch.ops.aten.reshape.default(sum_84, [196]);  sum_84 = None
    permute_198: "f32[196, 384]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_178: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_66, [8, 768, 384]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_312: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_313: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, add_47)
    mul_314: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_313, -0.5);  mul_313 = None
    exp_13: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_314);  mul_314 = None
    mul_315: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_316: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_47, mul_315);  add_47 = mul_315 = None
    add_150: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_312, mul_316);  mul_312 = mul_316 = None
    mul_317: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_178, add_150);  view_178 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_85: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1], True)
    view_179: "f32[384]" = torch.ops.aten.reshape.default(sum_85, [384]);  sum_85 = None
    view_180: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_317, [6144, 384]);  mul_317 = None
    permute_199: "f32[384, 6144]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_68: "f32[384, 196]" = torch.ops.aten.mm.default(permute_199, view_41);  permute_199 = view_41 = None
    permute_200: "f32[196, 384]" = torch.ops.aten.permute.default(mm_68, [1, 0]);  mm_68 = None
    mm_69: "f32[6144, 196]" = torch.ops.aten.mm.default(view_180, permute_201);  view_180 = permute_201 = None
    view_181: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_69, [8, 768, 196]);  mm_69 = None
    permute_202: "f32[384, 196]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_203: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_181, [0, 2, 1]);  view_181 = None
    clone_113: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    mul_319: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_113, primals_63);  primals_63 = None
    mul_320: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_319, 768)
    sum_86: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_319, mul_50);  mul_319 = None
    sum_87: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_87);  sum_87 = None
    sub_68: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_320, sum_86);  mul_320 = sum_86 = None
    sub_69: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_322);  sub_68 = mul_322 = None
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_69);  div_15 = sub_69 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_113, mul_50);  mul_50 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_113, [0, 1]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_148, mul_323);  add_148 = mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_151, [1568, 768])
    mm_70: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_182, permute_204);  permute_204 = None
    permute_205: "f32[768, 1568]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_71: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_205, view_39);  permute_205 = view_39 = None
    permute_206: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[768]" = torch.ops.aten.reshape.default(sum_90, [768]);  sum_90 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_184: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_70, [8, 196, 3072]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_326: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_43, 0.5);  add_43 = None
    mul_327: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_328: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_327, -0.5);  mul_327 = None
    exp_14: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_328);  mul_328 = None
    mul_329: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_330: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_329);  view_38 = mul_329 = None
    add_153: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_326, mul_330);  mul_326 = mul_330 = None
    mul_331: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_184, add_153);  view_184 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_331, [1568, 3072]);  mul_331 = None
    mm_72: "f32[1568, 768]" = torch.ops.aten.mm.default(view_185, permute_208);  permute_208 = None
    permute_209: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_185, [1, 0])
    mm_73: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_209, view_37);  permute_209 = view_37 = None
    permute_210: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[3072]" = torch.ops.aten.reshape.default(sum_91, [3072]);  sum_91 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_187: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_72, [8, 196, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_333: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_187, primals_57);  primals_57 = None
    mul_334: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_333, 768)
    sum_92: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [2], True)
    mul_335: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_45);  mul_333 = None
    sum_93: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True);  mul_335 = None
    mul_336: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_93);  sum_93 = None
    sub_71: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_334, sum_92);  mul_334 = sum_92 = None
    sub_72: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_336);  sub_71 = mul_336 = None
    mul_337: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_72);  div_16 = sub_72 = None
    mul_338: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_187, mul_45);  mul_45 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_338, [0, 1]);  mul_338 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_154: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_151, mul_337);  add_151 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_212: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_154, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_116: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    view_188: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_116, [6144, 196]);  clone_116 = None
    mm_74: "f32[6144, 384]" = torch.ops.aten.mm.default(view_188, permute_213);  permute_213 = None
    permute_214: "f32[196, 6144]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_75: "f32[196, 384]" = torch.ops.aten.mm.default(permute_214, view_35);  permute_214 = view_35 = None
    permute_215: "f32[384, 196]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_96: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[196]" = torch.ops.aten.reshape.default(sum_96, [196]);  sum_96 = None
    permute_216: "f32[196, 384]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_190: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_74, [8, 768, 384]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_340: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
    mul_341: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, add_38)
    mul_342: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_341, -0.5);  mul_341 = None
    exp_15: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_342);  mul_342 = None
    mul_343: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_344: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_38, mul_343);  add_38 = mul_343 = None
    add_156: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_340, mul_344);  mul_340 = mul_344 = None
    mul_345: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_190, add_156);  view_190 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_97: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1], True)
    view_191: "f32[384]" = torch.ops.aten.reshape.default(sum_97, [384]);  sum_97 = None
    view_192: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_345, [6144, 384]);  mul_345 = None
    permute_217: "f32[384, 6144]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_76: "f32[384, 196]" = torch.ops.aten.mm.default(permute_217, view_33);  permute_217 = view_33 = None
    permute_218: "f32[196, 384]" = torch.ops.aten.permute.default(mm_76, [1, 0]);  mm_76 = None
    mm_77: "f32[6144, 196]" = torch.ops.aten.mm.default(view_192, permute_219);  view_192 = permute_219 = None
    view_193: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_77, [8, 768, 196]);  mm_77 = None
    permute_220: "f32[384, 196]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_221: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    clone_117: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_117, primals_51);  primals_51 = None
    mul_348: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_347, 768)
    sum_98: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
    mul_349: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_347, mul_40);  mul_347 = None
    sum_99: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
    mul_350: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_99);  sum_99 = None
    sub_74: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_348, sum_98);  mul_348 = sum_98 = None
    sub_75: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_350);  sub_74 = mul_350 = None
    mul_351: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_75);  div_17 = sub_75 = None
    mul_352: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_117, mul_40);  mul_40 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_117, [0, 1]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_157: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_154, mul_351);  add_154 = mul_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_194: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_157, [1568, 768])
    mm_78: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_194, permute_222);  permute_222 = None
    permute_223: "f32[768, 1568]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_79: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_223, view_31);  permute_223 = view_31 = None
    permute_224: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_196: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_78, [8, 196, 3072]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_354: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_355: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, view_30)
    mul_356: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_355, -0.5);  mul_355 = None
    exp_16: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_356);  mul_356 = None
    mul_357: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_358: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_30, mul_357);  view_30 = mul_357 = None
    add_159: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_354, mul_358);  mul_354 = mul_358 = None
    mul_359: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_196, add_159);  view_196 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_359, [1568, 3072]);  mul_359 = None
    mm_80: "f32[1568, 768]" = torch.ops.aten.mm.default(view_197, permute_226);  permute_226 = None
    permute_227: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_81: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_227, view_29);  permute_227 = view_29 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[3072]" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_199: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_80, [8, 196, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_361: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_199, primals_45);  primals_45 = None
    mul_362: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_361, 768)
    sum_104: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
    mul_363: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_361, mul_35);  mul_361 = None
    sum_105: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
    mul_364: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_105);  sum_105 = None
    sub_77: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_362, sum_104);  mul_362 = sum_104 = None
    sub_78: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_364);  sub_77 = mul_364 = None
    mul_365: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_78);  div_18 = sub_78 = None
    mul_366: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_199, mul_35);  mul_35 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_160: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_157, mul_365);  add_157 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_230: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_160, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_120: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    view_200: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_120, [6144, 196]);  clone_120 = None
    mm_82: "f32[6144, 384]" = torch.ops.aten.mm.default(view_200, permute_231);  permute_231 = None
    permute_232: "f32[196, 6144]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_83: "f32[196, 384]" = torch.ops.aten.mm.default(permute_232, view_27);  permute_232 = view_27 = None
    permute_233: "f32[384, 196]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_108: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[196]" = torch.ops.aten.reshape.default(sum_108, [196]);  sum_108 = None
    permute_234: "f32[196, 384]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_202: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_82, [8, 768, 384]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_368: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_369: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, add_29)
    mul_370: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_369, -0.5);  mul_369 = None
    exp_17: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_370);  mul_370 = None
    mul_371: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_372: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_29, mul_371);  add_29 = mul_371 = None
    add_162: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_368, mul_372);  mul_368 = mul_372 = None
    mul_373: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_202, add_162);  view_202 = add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_109: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 1], True)
    view_203: "f32[384]" = torch.ops.aten.reshape.default(sum_109, [384]);  sum_109 = None
    view_204: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_373, [6144, 384]);  mul_373 = None
    permute_235: "f32[384, 6144]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_84: "f32[384, 196]" = torch.ops.aten.mm.default(permute_235, view_25);  permute_235 = view_25 = None
    permute_236: "f32[196, 384]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    mm_85: "f32[6144, 196]" = torch.ops.aten.mm.default(view_204, permute_237);  view_204 = permute_237 = None
    view_205: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_85, [8, 768, 196]);  mm_85 = None
    permute_238: "f32[384, 196]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_239: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    clone_121: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    mul_375: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_121, primals_39);  primals_39 = None
    mul_376: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_375, 768)
    sum_110: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True)
    mul_377: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_375, mul_30);  mul_375 = None
    sum_111: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
    mul_378: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_111);  sum_111 = None
    sub_80: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_376, sum_110);  mul_376 = sum_110 = None
    sub_81: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_378);  sub_80 = mul_378 = None
    mul_379: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_81);  div_19 = sub_81 = None
    mul_380: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_121, mul_30);  mul_30 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_380, [0, 1]);  mul_380 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_121, [0, 1]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_163: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_160, mul_379);  add_160 = mul_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_206: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_163, [1568, 768])
    mm_86: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_206, permute_240);  permute_240 = None
    permute_241: "f32[768, 1568]" = torch.ops.aten.permute.default(view_206, [1, 0])
    mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_23);  permute_241 = view_23 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
    view_207: "f32[768]" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
    permute_243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_208: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_86, [8, 196, 3072]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_382: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_25, 0.5);  add_25 = None
    mul_383: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, view_22)
    mul_384: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_383, -0.5);  mul_383 = None
    exp_18: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_384);  mul_384 = None
    mul_385: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_386: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_22, mul_385);  view_22 = mul_385 = None
    add_165: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_382, mul_386);  mul_382 = mul_386 = None
    mul_387: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_208, add_165);  view_208 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_209: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_387, [1568, 3072]);  mul_387 = None
    mm_88: "f32[1568, 768]" = torch.ops.aten.mm.default(view_209, permute_244);  permute_244 = None
    permute_245: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_21);  permute_245 = view_21 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[3072]" = torch.ops.aten.reshape.default(sum_115, [3072]);  sum_115 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_211: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_88, [8, 196, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_389: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, primals_33);  primals_33 = None
    mul_390: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_389, 768)
    sum_116: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True)
    mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_389, mul_25);  mul_389 = None
    sum_117: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True);  mul_391 = None
    mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_117);  sum_117 = None
    sub_83: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_390, sum_116);  mul_390 = sum_116 = None
    sub_84: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_392);  sub_83 = mul_392 = None
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_84);  div_20 = sub_84 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_25);  mul_25 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_394, [0, 1]);  mul_394 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_166: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_163, mul_393);  add_163 = mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_248: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_166, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_124: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
    view_212: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_124, [6144, 196]);  clone_124 = None
    mm_90: "f32[6144, 384]" = torch.ops.aten.mm.default(view_212, permute_249);  permute_249 = None
    permute_250: "f32[196, 6144]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_91: "f32[196, 384]" = torch.ops.aten.mm.default(permute_250, view_19);  permute_250 = view_19 = None
    permute_251: "f32[384, 196]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_120: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[196]" = torch.ops.aten.reshape.default(sum_120, [196]);  sum_120 = None
    permute_252: "f32[196, 384]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_214: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_90, [8, 768, 384]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_396: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_21, 0.5);  add_21 = None
    mul_397: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, add_20)
    mul_398: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_397, -0.5);  mul_397 = None
    exp_19: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_398);  mul_398 = None
    mul_399: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_400: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_20, mul_399);  add_20 = mul_399 = None
    add_168: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_396, mul_400);  mul_396 = mul_400 = None
    mul_401: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_214, add_168);  view_214 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_121: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1], True)
    view_215: "f32[384]" = torch.ops.aten.reshape.default(sum_121, [384]);  sum_121 = None
    view_216: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_401, [6144, 384]);  mul_401 = None
    permute_253: "f32[384, 6144]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_92: "f32[384, 196]" = torch.ops.aten.mm.default(permute_253, view_17);  permute_253 = view_17 = None
    permute_254: "f32[196, 384]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    mm_93: "f32[6144, 196]" = torch.ops.aten.mm.default(view_216, permute_255);  view_216 = permute_255 = None
    view_217: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_93, [8, 768, 196]);  mm_93 = None
    permute_256: "f32[384, 196]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_257: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    clone_125: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    mul_403: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_125, primals_27);  primals_27 = None
    mul_404: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_403, 768)
    sum_122: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True)
    mul_405: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_403, mul_20);  mul_403 = None
    sum_123: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True);  mul_405 = None
    mul_406: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, sum_123);  sum_123 = None
    sub_86: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_404, sum_122);  mul_404 = sum_122 = None
    sub_87: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_406);  sub_86 = mul_406 = None
    mul_407: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_87);  div_21 = sub_87 = None
    mul_408: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_125, mul_20);  mul_20 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1]);  mul_408 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_125, [0, 1]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_169: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_166, mul_407);  add_166 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_218: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_169, [1568, 768])
    mm_94: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_218, permute_258);  permute_258 = None
    permute_259: "f32[768, 1568]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_95: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_259, view_15);  permute_259 = view_15 = None
    permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_220: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_94, [8, 196, 3072]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_410: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_411: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, view_14)
    mul_412: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_411, -0.5);  mul_411 = None
    exp_20: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_412);  mul_412 = None
    mul_413: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_414: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_14, mul_413);  view_14 = mul_413 = None
    add_171: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_410, mul_414);  mul_410 = mul_414 = None
    mul_415: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_220, add_171);  view_220 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_221: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_415, [1568, 3072]);  mul_415 = None
    mm_96: "f32[1568, 768]" = torch.ops.aten.mm.default(view_221, permute_262);  permute_262 = None
    permute_263: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_97: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_263, view_13);  permute_263 = view_13 = None
    permute_264: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_127: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
    view_222: "f32[3072]" = torch.ops.aten.reshape.default(sum_127, [3072]);  sum_127 = None
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_223: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_96, [8, 196, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_223, primals_21);  primals_21 = None
    mul_418: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_417, 768)
    sum_128: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True)
    mul_419: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_417, mul_15);  mul_417 = None
    sum_129: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [2], True);  mul_419 = None
    mul_420: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_129);  sum_129 = None
    sub_89: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_418, sum_128);  mul_418 = sum_128 = None
    sub_90: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_420);  sub_89 = mul_420 = None
    mul_421: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_90);  div_22 = sub_90 = None
    mul_422: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_223, mul_15);  mul_15 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 1]);  mul_422 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_223, [0, 1]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_169, mul_421);  add_169 = mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_266: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_172, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_128: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_266, memory_format = torch.contiguous_format);  permute_266 = None
    view_224: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_128, [6144, 196]);  clone_128 = None
    mm_98: "f32[6144, 384]" = torch.ops.aten.mm.default(view_224, permute_267);  permute_267 = None
    permute_268: "f32[196, 6144]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_99: "f32[196, 384]" = torch.ops.aten.mm.default(permute_268, view_11);  permute_268 = view_11 = None
    permute_269: "f32[384, 196]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_132: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[196]" = torch.ops.aten.reshape.default(sum_132, [196]);  sum_132 = None
    permute_270: "f32[196, 384]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_226: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_98, [8, 768, 384]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_424: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
    mul_425: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, add_11)
    mul_426: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_425, -0.5);  mul_425 = None
    exp_21: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_426);  mul_426 = None
    mul_427: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_428: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_11, mul_427);  add_11 = mul_427 = None
    add_174: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_424, mul_428);  mul_424 = mul_428 = None
    mul_429: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_226, add_174);  view_226 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_133: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 1], True)
    view_227: "f32[384]" = torch.ops.aten.reshape.default(sum_133, [384]);  sum_133 = None
    view_228: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_429, [6144, 384]);  mul_429 = None
    permute_271: "f32[384, 6144]" = torch.ops.aten.permute.default(view_228, [1, 0])
    mm_100: "f32[384, 196]" = torch.ops.aten.mm.default(permute_271, view_9);  permute_271 = view_9 = None
    permute_272: "f32[196, 384]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    mm_101: "f32[6144, 196]" = torch.ops.aten.mm.default(view_228, permute_273);  view_228 = permute_273 = None
    view_229: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_101, [8, 768, 196]);  mm_101 = None
    permute_274: "f32[384, 196]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_275: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    clone_129: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    mul_431: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_129, primals_15);  primals_15 = None
    mul_432: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_431, 768)
    sum_134: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True)
    mul_433: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_431, mul_10);  mul_431 = None
    sum_135: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True);  mul_433 = None
    mul_434: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_135);  sum_135 = None
    sub_92: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_432, sum_134);  mul_432 = sum_134 = None
    sub_93: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_434);  sub_92 = mul_434 = None
    mul_435: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_93);  div_23 = sub_93 = None
    mul_436: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_129, mul_10);  mul_10 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1]);  mul_436 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_129, [0, 1]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_175: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_172, mul_435);  add_172 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_230: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_175, [1568, 768])
    mm_102: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_230, permute_276);  permute_276 = None
    permute_277: "f32[768, 1568]" = torch.ops.aten.permute.default(view_230, [1, 0])
    mm_103: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_277, view_7);  permute_277 = view_7 = None
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
    permute_279: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_102, [8, 196, 3072]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_438: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_439: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, view_6)
    mul_440: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_439, -0.5);  mul_439 = None
    exp_22: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_440);  mul_440 = None
    mul_441: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_442: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_6, mul_441);  view_6 = mul_441 = None
    add_177: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_438, mul_442);  mul_438 = mul_442 = None
    mul_443: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, add_177);  view_232 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_233: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_443, [1568, 3072]);  mul_443 = None
    mm_104: "f32[1568, 768]" = torch.ops.aten.mm.default(view_233, permute_280);  permute_280 = None
    permute_281: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_105: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_281, view_5);  permute_281 = view_5 = None
    permute_282: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[3072]" = torch.ops.aten.reshape.default(sum_139, [3072]);  sum_139 = None
    permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_235: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_104, [8, 196, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    mul_445: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_235, primals_9);  primals_9 = None
    mul_446: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_445, 768)
    sum_140: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
    mul_447: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_445, mul_5);  mul_445 = None
    sum_141: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    mul_448: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, sum_141);  sum_141 = None
    sub_95: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_446, sum_140);  mul_446 = sum_140 = None
    sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_448);  sub_95 = mul_448 = None
    mul_449: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_96);  div_24 = sub_96 = None
    mul_450: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_235, mul_5);  mul_5 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:81, code: x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
    add_178: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_175, mul_449);  add_175 = mul_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_284: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_178, [0, 2, 1])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_132: "f32[8, 768, 196]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
    view_236: "f32[6144, 196]" = torch.ops.aten.reshape.default(clone_132, [6144, 196]);  clone_132 = None
    mm_106: "f32[6144, 384]" = torch.ops.aten.mm.default(view_236, permute_285);  permute_285 = None
    permute_286: "f32[196, 6144]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_107: "f32[196, 384]" = torch.ops.aten.mm.default(permute_286, view_3);  permute_286 = view_3 = None
    permute_287: "f32[384, 196]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_144: "f32[1, 196]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[196]" = torch.ops.aten.reshape.default(sum_144, [196]);  sum_144 = None
    permute_288: "f32[196, 384]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    view_238: "f32[8, 768, 384]" = torch.ops.aten.reshape.default(mm_106, [8, 768, 384]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_452: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
    mul_453: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, add_2)
    mul_454: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(mul_453, -0.5);  mul_453 = None
    exp_23: "f32[8, 768, 384]" = torch.ops.aten.exp.default(mul_454);  mul_454 = None
    mul_455: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_456: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(add_2, mul_455);  add_2 = mul_455 = None
    add_180: "f32[8, 768, 384]" = torch.ops.aten.add.Tensor(mul_452, mul_456);  mul_452 = mul_456 = None
    mul_457: "f32[8, 768, 384]" = torch.ops.aten.mul.Tensor(view_238, add_180);  view_238 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    sum_145: "f32[1, 1, 384]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 1], True)
    view_239: "f32[384]" = torch.ops.aten.reshape.default(sum_145, [384]);  sum_145 = None
    view_240: "f32[6144, 384]" = torch.ops.aten.reshape.default(mul_457, [6144, 384]);  mul_457 = None
    permute_289: "f32[384, 6144]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_108: "f32[384, 196]" = torch.ops.aten.mm.default(permute_289, view_1);  permute_289 = view_1 = None
    permute_290: "f32[196, 384]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    mm_109: "f32[6144, 196]" = torch.ops.aten.mm.default(view_240, permute_291);  view_240 = permute_291 = None
    view_241: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(mm_109, [8, 768, 196]);  mm_109 = None
    permute_292: "f32[384, 196]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    permute_293: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view_241, [0, 2, 1]);  view_241 = None
    clone_133: "f32[8, 196, 768]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    mul_459: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_133, primals_3);  primals_3 = None
    mul_460: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, 768)
    sum_146: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_459, mul);  mul_459 = None
    sum_147: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, sum_147);  sum_147 = None
    sub_98: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_460, sum_146);  mul_460 = sum_146 = None
    sub_99: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_462);  sub_98 = mul_462 = None
    mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_99);  div_25 = sub_99 = None
    mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(clone_133, mul);  mul = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_133, [0, 1]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/mlp_mixer.py:80, code: x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
    add_181: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_178, mul_463);  add_178 = mul_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_294: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_181, [0, 2, 1]);  add_181 = None
    view_242: "f32[8, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_294, [8, 768, 14, 14]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_242, primals_151, primals_1, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_242 = primals_151 = primals_1 = None
    getitem_51: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_52: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return [getitem_51, getitem_52, sum_148, sum_149, permute_292, view_239, permute_288, view_237, sum_142, sum_143, permute_283, view_234, permute_279, view_231, sum_136, sum_137, permute_274, view_227, permute_270, view_225, sum_130, sum_131, permute_265, view_222, permute_261, view_219, sum_124, sum_125, permute_256, view_215, permute_252, view_213, sum_118, sum_119, permute_247, view_210, permute_243, view_207, sum_112, sum_113, permute_238, view_203, permute_234, view_201, sum_106, sum_107, permute_229, view_198, permute_225, view_195, sum_100, sum_101, permute_220, view_191, permute_216, view_189, sum_94, sum_95, permute_211, view_186, permute_207, view_183, sum_88, sum_89, permute_202, view_179, permute_198, view_177, sum_82, sum_83, permute_193, view_174, permute_189, view_171, sum_76, sum_77, permute_184, view_167, permute_180, view_165, sum_70, sum_71, permute_175, view_162, permute_171, view_159, sum_64, sum_65, permute_166, view_155, permute_162, view_153, sum_58, sum_59, permute_157, view_150, permute_153, view_147, sum_52, sum_53, permute_148, view_143, permute_144, view_141, sum_46, sum_47, permute_139, view_138, permute_135, view_135, sum_40, sum_41, permute_130, view_131, permute_126, view_129, sum_34, sum_35, permute_121, view_126, permute_117, view_123, sum_28, sum_29, permute_112, view_119, permute_108, view_117, sum_22, sum_23, permute_103, view_114, permute_99, view_111, sum_16, sum_17, permute_94, view_107, permute_90, view_105, sum_10, sum_11, permute_85, view_102, permute_81, view_99, sum_4, sum_5, permute_77, view_97, None]
    