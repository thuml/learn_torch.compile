from __future__ import annotations



def forward(self, primals_2: "f32[768]", primals_3: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_109: "f32[768]", primals_110: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_122: "f32[768]", primals_124: "f32[768, 3, 16, 16]", primals_224: "f32[8, 3, 224, 224]", cat: "f32[8, 197, 768]", getitem_1: "f32[8, 197, 1]", rsqrt: "f32[8, 197, 1]", view_1: "f32[1576, 768]", getitem_2: "f32[8, 12, 197, 64]", getitem_3: "f32[8, 12, 197, 64]", getitem_4: "f32[8, 12, 197, 64]", view_4: "i64[38809]", expand_1: "f32[8, 12, 197, 197]", getitem_6: "f32[8, 12, 224]", getitem_7: "i64[]", getitem_8: "i64[]", view_7: "f32[1576, 768]", addmm_1: "f32[1576, 768]", mul_3: "f32[8, 197, 768]", view_9: "f32[1576, 768]", addmm_2: "f32[1576, 3072]", view_11: "f32[1576, 3072]", addmm_3: "f32[1576, 768]", mul_9: "f32[8, 197, 768]", view_13: "f32[1576, 768]", getitem_13: "f32[8, 12, 197, 64]", getitem_14: "f32[8, 12, 197, 64]", getitem_15: "f32[8, 12, 197, 64]", view_16: "i64[38809]", expand_2: "f32[8, 12, 197, 197]", getitem_17: "f32[8, 12, 224]", getitem_18: "i64[]", getitem_19: "i64[]", view_19: "f32[1576, 768]", addmm_5: "f32[1576, 768]", mul_12: "f32[8, 197, 768]", view_21: "f32[1576, 768]", addmm_6: "f32[1576, 3072]", view_23: "f32[1576, 3072]", addmm_7: "f32[1576, 768]", mul_18: "f32[8, 197, 768]", view_25: "f32[1576, 768]", getitem_24: "f32[8, 12, 197, 64]", getitem_25: "f32[8, 12, 197, 64]", getitem_26: "f32[8, 12, 197, 64]", view_28: "i64[38809]", expand_3: "f32[8, 12, 197, 197]", getitem_28: "f32[8, 12, 224]", getitem_29: "i64[]", getitem_30: "i64[]", view_31: "f32[1576, 768]", addmm_9: "f32[1576, 768]", mul_21: "f32[8, 197, 768]", view_33: "f32[1576, 768]", addmm_10: "f32[1576, 3072]", view_35: "f32[1576, 3072]", addmm_11: "f32[1576, 768]", mul_27: "f32[8, 197, 768]", view_37: "f32[1576, 768]", getitem_35: "f32[8, 12, 197, 64]", getitem_36: "f32[8, 12, 197, 64]", getitem_37: "f32[8, 12, 197, 64]", view_40: "i64[38809]", expand_4: "f32[8, 12, 197, 197]", getitem_39: "f32[8, 12, 224]", getitem_40: "i64[]", getitem_41: "i64[]", view_43: "f32[1576, 768]", addmm_13: "f32[1576, 768]", mul_30: "f32[8, 197, 768]", view_45: "f32[1576, 768]", addmm_14: "f32[1576, 3072]", view_47: "f32[1576, 3072]", addmm_15: "f32[1576, 768]", mul_36: "f32[8, 197, 768]", view_49: "f32[1576, 768]", getitem_46: "f32[8, 12, 197, 64]", getitem_47: "f32[8, 12, 197, 64]", getitem_48: "f32[8, 12, 197, 64]", view_52: "i64[38809]", expand_5: "f32[8, 12, 197, 197]", getitem_50: "f32[8, 12, 224]", getitem_51: "i64[]", getitem_52: "i64[]", view_55: "f32[1576, 768]", addmm_17: "f32[1576, 768]", mul_39: "f32[8, 197, 768]", view_57: "f32[1576, 768]", addmm_18: "f32[1576, 3072]", view_59: "f32[1576, 3072]", addmm_19: "f32[1576, 768]", mul_45: "f32[8, 197, 768]", view_61: "f32[1576, 768]", getitem_57: "f32[8, 12, 197, 64]", getitem_58: "f32[8, 12, 197, 64]", getitem_59: "f32[8, 12, 197, 64]", view_64: "i64[38809]", expand_6: "f32[8, 12, 197, 197]", getitem_61: "f32[8, 12, 224]", getitem_62: "i64[]", getitem_63: "i64[]", view_67: "f32[1576, 768]", addmm_21: "f32[1576, 768]", mul_48: "f32[8, 197, 768]", view_69: "f32[1576, 768]", addmm_22: "f32[1576, 3072]", view_71: "f32[1576, 3072]", addmm_23: "f32[1576, 768]", mul_54: "f32[8, 197, 768]", view_73: "f32[1576, 768]", getitem_68: "f32[8, 12, 197, 64]", getitem_69: "f32[8, 12, 197, 64]", getitem_70: "f32[8, 12, 197, 64]", view_76: "i64[38809]", expand_7: "f32[8, 12, 197, 197]", getitem_72: "f32[8, 12, 224]", getitem_73: "i64[]", getitem_74: "i64[]", view_79: "f32[1576, 768]", addmm_25: "f32[1576, 768]", mul_57: "f32[8, 197, 768]", view_81: "f32[1576, 768]", addmm_26: "f32[1576, 3072]", view_83: "f32[1576, 3072]", addmm_27: "f32[1576, 768]", mul_63: "f32[8, 197, 768]", view_85: "f32[1576, 768]", getitem_79: "f32[8, 12, 197, 64]", getitem_80: "f32[8, 12, 197, 64]", getitem_81: "f32[8, 12, 197, 64]", view_88: "i64[38809]", expand_8: "f32[8, 12, 197, 197]", getitem_83: "f32[8, 12, 224]", getitem_84: "i64[]", getitem_85: "i64[]", view_91: "f32[1576, 768]", addmm_29: "f32[1576, 768]", mul_66: "f32[8, 197, 768]", view_93: "f32[1576, 768]", addmm_30: "f32[1576, 3072]", view_95: "f32[1576, 3072]", addmm_31: "f32[1576, 768]", mul_72: "f32[8, 197, 768]", view_97: "f32[1576, 768]", getitem_90: "f32[8, 12, 197, 64]", getitem_91: "f32[8, 12, 197, 64]", getitem_92: "f32[8, 12, 197, 64]", view_100: "i64[38809]", expand_9: "f32[8, 12, 197, 197]", getitem_94: "f32[8, 12, 224]", getitem_95: "i64[]", getitem_96: "i64[]", view_103: "f32[1576, 768]", addmm_33: "f32[1576, 768]", mul_75: "f32[8, 197, 768]", view_105: "f32[1576, 768]", addmm_34: "f32[1576, 3072]", view_107: "f32[1576, 3072]", addmm_35: "f32[1576, 768]", mul_81: "f32[8, 197, 768]", view_109: "f32[1576, 768]", getitem_101: "f32[8, 12, 197, 64]", getitem_102: "f32[8, 12, 197, 64]", getitem_103: "f32[8, 12, 197, 64]", view_112: "i64[38809]", expand_10: "f32[8, 12, 197, 197]", getitem_105: "f32[8, 12, 224]", getitem_106: "i64[]", getitem_107: "i64[]", view_115: "f32[1576, 768]", addmm_37: "f32[1576, 768]", mul_84: "f32[8, 197, 768]", view_117: "f32[1576, 768]", addmm_38: "f32[1576, 3072]", view_119: "f32[1576, 3072]", addmm_39: "f32[1576, 768]", mul_90: "f32[8, 197, 768]", view_121: "f32[1576, 768]", getitem_112: "f32[8, 12, 197, 64]", getitem_113: "f32[8, 12, 197, 64]", getitem_114: "f32[8, 12, 197, 64]", view_124: "i64[38809]", expand_11: "f32[8, 12, 197, 197]", getitem_116: "f32[8, 12, 224]", getitem_117: "i64[]", getitem_118: "i64[]", view_127: "f32[1576, 768]", addmm_41: "f32[1576, 768]", mul_93: "f32[8, 197, 768]", view_129: "f32[1576, 768]", addmm_42: "f32[1576, 3072]", view_131: "f32[1576, 3072]", addmm_43: "f32[1576, 768]", mul_99: "f32[8, 197, 768]", view_133: "f32[1576, 768]", getitem_123: "f32[8, 12, 197, 64]", getitem_124: "f32[8, 12, 197, 64]", getitem_125: "f32[8, 12, 197, 64]", view_136: "i64[38809]", expand_12: "f32[8, 12, 197, 197]", getitem_127: "f32[8, 12, 224]", getitem_128: "i64[]", getitem_129: "i64[]", view_139: "f32[1576, 768]", addmm_45: "f32[1576, 768]", mul_102: "f32[8, 197, 768]", view_141: "f32[1576, 768]", addmm_46: "f32[1576, 3072]", view_143: "f32[1576, 3072]", addmm_47: "f32[1576, 768]", mul_108: "f32[8, 768]", clone_49: "f32[8, 768]", permute_86: "f32[1000, 768]", div: "f32[8, 1]", permute_90: "f32[768, 3072]", permute_94: "f32[3072, 768]", div_2: "f32[8, 197, 1]", permute_98: "f32[768, 768]", alias_12: "f32[8, 12, 197, 64]", permute_105: "f32[2304, 768]", div_3: "f32[8, 197, 1]", permute_109: "f32[768, 3072]", permute_113: "f32[3072, 768]", div_4: "f32[8, 197, 1]", permute_117: "f32[768, 768]", alias_13: "f32[8, 12, 197, 64]", permute_124: "f32[2304, 768]", div_5: "f32[8, 197, 1]", permute_128: "f32[768, 3072]", permute_132: "f32[3072, 768]", div_6: "f32[8, 197, 1]", permute_136: "f32[768, 768]", alias_14: "f32[8, 12, 197, 64]", permute_143: "f32[2304, 768]", div_7: "f32[8, 197, 1]", permute_147: "f32[768, 3072]", permute_151: "f32[3072, 768]", div_8: "f32[8, 197, 1]", permute_155: "f32[768, 768]", alias_15: "f32[8, 12, 197, 64]", permute_162: "f32[2304, 768]", div_9: "f32[8, 197, 1]", permute_166: "f32[768, 3072]", permute_170: "f32[3072, 768]", div_10: "f32[8, 197, 1]", permute_174: "f32[768, 768]", alias_16: "f32[8, 12, 197, 64]", permute_181: "f32[2304, 768]", div_11: "f32[8, 197, 1]", permute_185: "f32[768, 3072]", permute_189: "f32[3072, 768]", div_12: "f32[8, 197, 1]", permute_193: "f32[768, 768]", alias_17: "f32[8, 12, 197, 64]", permute_200: "f32[2304, 768]", div_13: "f32[8, 197, 1]", permute_204: "f32[768, 3072]", permute_208: "f32[3072, 768]", div_14: "f32[8, 197, 1]", permute_212: "f32[768, 768]", alias_18: "f32[8, 12, 197, 64]", permute_219: "f32[2304, 768]", div_15: "f32[8, 197, 1]", permute_223: "f32[768, 3072]", permute_227: "f32[3072, 768]", div_16: "f32[8, 197, 1]", permute_231: "f32[768, 768]", alias_19: "f32[8, 12, 197, 64]", permute_238: "f32[2304, 768]", div_17: "f32[8, 197, 1]", permute_242: "f32[768, 3072]", permute_246: "f32[3072, 768]", div_18: "f32[8, 197, 1]", permute_250: "f32[768, 768]", alias_20: "f32[8, 12, 197, 64]", permute_257: "f32[2304, 768]", div_19: "f32[8, 197, 1]", permute_261: "f32[768, 3072]", permute_265: "f32[3072, 768]", div_20: "f32[8, 197, 1]", permute_269: "f32[768, 768]", alias_21: "f32[8, 12, 197, 64]", permute_276: "f32[2304, 768]", div_21: "f32[8, 197, 1]", permute_280: "f32[768, 3072]", permute_284: "f32[3072, 768]", div_22: "f32[8, 197, 1]", permute_288: "f32[768, 768]", alias_22: "f32[8, 12, 197, 64]", permute_295: "f32[2304, 768]", div_23: "f32[8, 197, 1]", permute_299: "f32[768, 3072]", permute_303: "f32[3072, 768]", div_24: "f32[8, 197, 1]", permute_307: "f32[768, 768]", alias_23: "f32[8, 12, 197, 64]", permute_314: "f32[2304, 768]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_1);  cat = getitem_1 = None
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_8: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_10: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_6: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.7071067811865476)
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_12: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_20: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_15: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476)
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_12: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_32: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_24: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476)
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_19: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_36: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_44: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_46: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476)
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_26: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_48: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_56: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_33: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_60: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_68: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476)
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_40: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_80: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_82: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476)
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_47: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_84: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_92: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_94: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_69: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476)
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_96: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_104: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_78: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_61: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_116: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_87: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_68: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_120: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_128: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_130: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_75: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_132: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_140: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_82: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_144: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_86);  permute_86 = None
    permute_87: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_87, clone_49);  permute_87 = clone_49 = None
    permute_88: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_145: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_89: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_111: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, primals_122);  primals_122 = None
    mul_112: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_111, 768)
    sum_2: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [1], True)
    mul_113: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_111, mul_108);  mul_111 = None
    sum_3: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [1], True);  mul_113 = None
    mul_114: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_3);  sum_3 = None
    sub_26: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mul_112, sum_2);  mul_112 = sum_2 = None
    sub_27: "f32[8, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_114);  sub_26 = mul_114 = None
    mul_115: "f32[8, 768]" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
    mul_116: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, mul_108);  mul_108 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_116, [0]);  mul_116 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm, [0]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    unsqueeze_12: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_115, 1);  mul_115 = None
    expand_13: "f32[8, 196, 768]" = torch.ops.aten.expand.default(unsqueeze_12, [8, 196, 768]);  unsqueeze_12 = None
    div_1: "f32[8, 196, 768]" = torch.ops.aten.div.Scalar(expand_13, 196);  expand_13 = None
    full_default: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, div_1, 1, 1, 9223372036854775807);  full_default = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_117: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_119);  primals_119 = None
    mul_118: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, view_144);  view_144 = None
    sum_6: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_118, [0, 1], True);  mul_118 = None
    view_146: "f32[768]" = torch.ops.aten.reshape.default(sum_6, [768]);  sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_117, [1576, 768]);  mul_117 = None
    mm_2: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_147, permute_90);  permute_90 = None
    permute_91: "f32[768, 1576]" = torch.ops.aten.permute.default(view_147, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_91, view_143);  permute_91 = view_143 = None
    permute_92: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
    view_148: "f32[768]" = torch.ops.aten.reshape.default(sum_7, [768]);  sum_7 = None
    permute_93: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    view_149: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_2, [8, 197, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_120: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
    mul_121: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, view_142)
    mul_122: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_121, -0.5);  mul_121 = None
    exp: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_122);  mul_122 = None
    mul_123: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_124: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, mul_123);  view_142 = mul_123 = None
    add_87: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_120, mul_124);  mul_120 = mul_124 = None
    mul_125: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_149, add_87);  view_149 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_150: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_125, [1576, 3072]);  mul_125 = None
    mm_4: "f32[1576, 768]" = torch.ops.aten.mm.default(view_150, permute_94);  permute_94 = None
    permute_95: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_150, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_95, view_141);  permute_95 = view_141 = None
    permute_96: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_8: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
    view_151: "f32[3072]" = torch.ops.aten.reshape.default(sum_8, [3072]);  sum_8 = None
    permute_97: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    view_152: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_4, [8, 197, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_152, primals_120);  primals_120 = None
    mul_128: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_127, 768)
    sum_9: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
    mul_129: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_127, mul_102);  mul_127 = None
    sum_10: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
    mul_130: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_102, sum_10);  sum_10 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_128, sum_9);  mul_128 = sum_9 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_130);  sub_29 = mul_130 = None
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_30);  div_2 = sub_30 = None
    mul_132: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_102);  mul_102 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_88: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(slice_scatter, mul_131);  slice_scatter = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_133: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_88, primals_112);  primals_112 = None
    mul_134: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_88, view_140);  view_140 = None
    sum_13: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1], True);  mul_134 = None
    view_153: "f32[768]" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_154: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_133, [1576, 768]);  mul_133 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_154, permute_98);  permute_98 = None
    permute_99: "f32[768, 1576]" = torch.ops.aten.permute.default(view_154, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_99, view_139);  permute_99 = view_139 = None
    permute_100: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_154, [0], True);  view_154 = None
    view_155: "f32[768]" = torch.ops.aten.reshape.default(sum_14, [768]);  sum_14 = None
    permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    view_156: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 197, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_157: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_156, [8, 197, 12, 64]);  view_156 = None
    permute_102: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_102, getitem_123, getitem_124, getitem_125, expand_12, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, True]);  permute_102 = getitem_123 = getitem_124 = getitem_125 = expand_12 = alias_12 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_134: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_135: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_136: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward[2]
    getitem_137: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward[3];  _scaled_dot_product_efficient_attention_backward = None
    sum_15: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_137, [0], True);  getitem_137 = None
    full_default_2: "f32[1, 12, 197, 200]" = torch.ops.aten.full.default([1, 12, 197, 200], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_2: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_15, -1, 0, 197);  sum_15 = None
    constant_pad_nd_12: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_2, [0, -3]);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_12, 0);  constant_pad_nd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_103: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_158: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_103, [38809, 12]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_default_3: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    index_put: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_136], view_158, True);  view_136 = view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_13: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_134, getitem_135, getitem_136]);  getitem_134 = getitem_135 = getitem_136 = None
    view_159: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_13, [3, 8, 12, 197, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_104: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_159, [1, 3, 0, 2, 4]);  view_159 = None
    clone_50: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_160: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_50, [8, 197, 2304]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_161: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_160, [1576, 2304]);  view_160 = None
    mm_8: "f32[1576, 768]" = torch.ops.aten.mm.default(view_161, permute_105);  permute_105 = None
    permute_106: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_161, [1, 0])
    mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_106, view_133);  permute_106 = view_133 = None
    permute_107: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_16: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_161, [0], True);  view_161 = None
    view_162: "f32[2304]" = torch.ops.aten.reshape.default(sum_16, [2304]);  sum_16 = None
    permute_108: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_163: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_8, [8, 197, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_15: "f32[768]" = torch.ops.aten.slice.Tensor(view_162, 0, 0, 768)
    slice_17: "f32[768]" = torch.ops.aten.slice.Tensor(view_162, 0, 1536, 2304);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_136: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_163, primals_113);  primals_113 = None
    mul_137: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_136, 768)
    sum_17: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [2], True)
    mul_138: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_136, mul_99);  mul_136 = None
    sum_18: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
    mul_139: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, sum_18);  sum_18 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_137, sum_17);  mul_137 = sum_17 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_139);  sub_32 = mul_139 = None
    mul_140: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_33);  div_3 = sub_33 = None
    mul_141: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_163, mul_99);  mul_99 = None
    sum_19: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1]);  mul_141 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_163, [0, 1]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_88, mul_140);  add_88 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_89, primals_109);  primals_109 = None
    mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_89, view_132);  view_132 = None
    sum_21: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1], True);  mul_143 = None
    view_164: "f32[768]" = torch.ops.aten.reshape.default(sum_21, [768]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_165: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_142, [1576, 768]);  mul_142 = None
    mm_10: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_165, permute_109);  permute_109 = None
    permute_110: "f32[768, 1576]" = torch.ops.aten.permute.default(view_165, [1, 0])
    mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_110, view_131);  permute_110 = view_131 = None
    permute_111: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
    view_166: "f32[768]" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
    permute_112: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_167: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_10, [8, 197, 3072]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_145: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_75, 0.5);  add_75 = None
    mul_146: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, view_130)
    mul_147: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_146, -0.5);  mul_146 = None
    exp_1: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_147);  mul_147 = None
    mul_148: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_149: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, mul_148);  view_130 = mul_148 = None
    add_91: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_145, mul_149);  mul_145 = mul_149 = None
    mul_150: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_167, add_91);  view_167 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_168: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_150, [1576, 3072]);  mul_150 = None
    mm_12: "f32[1576, 768]" = torch.ops.aten.mm.default(view_168, permute_113);  permute_113 = None
    permute_114: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_114, view_129);  permute_114 = view_129 = None
    permute_115: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_23: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[3072]" = torch.ops.aten.reshape.default(sum_23, [3072]);  sum_23 = None
    permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    view_170: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_12, [8, 197, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_170, primals_110);  primals_110 = None
    mul_153: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_152, 768)
    sum_24: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_152, mul_93);  mul_152 = None
    sum_25: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, sum_25);  sum_25 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_153, sum_24);  mul_153 = sum_24 = None
    sub_36: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_155);  sub_35 = mul_155 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_36);  div_4 = sub_36 = None
    mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_170, mul_93);  mul_93 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_92: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_89, mul_156);  add_89 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_92, primals_102);  primals_102 = None
    mul_159: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_92, view_128);  view_128 = None
    sum_28: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1], True);  mul_159 = None
    view_171: "f32[768]" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_172: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_158, [1576, 768]);  mul_158 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_172, permute_117);  permute_117 = None
    permute_118: "f32[768, 1576]" = torch.ops.aten.permute.default(view_172, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_118, view_127);  permute_118 = view_127 = None
    permute_119: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
    view_173: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
    permute_120: "f32[768, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    view_174: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 197, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_175: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_174, [8, 197, 12, 64]);  view_174 = None
    permute_121: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_121, getitem_112, getitem_113, getitem_114, expand_11, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, True]);  permute_121 = getitem_112 = getitem_113 = getitem_114 = expand_11 = alias_13 = getitem_116 = getitem_117 = getitem_118 = None
    getitem_138: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_139: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_140: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_1[2]
    getitem_141: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_1[3];  _scaled_dot_product_efficient_attention_backward_1 = None
    sum_30: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_141, [0], True);  getitem_141 = None
    slice_scatter_3: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_30, -1, 0, 197);  sum_30 = None
    constant_pad_nd_13: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_3, [0, -3]);  slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_13, 0);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_122: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_176: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_122, [38809, 12]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_124], view_176, True);  view_124 = view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_14: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140]);  getitem_138 = getitem_139 = getitem_140 = None
    view_177: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_14, [3, 8, 12, 197, 64]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_123: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_177, [1, 3, 0, 2, 4]);  view_177 = None
    clone_51: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    view_178: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_51, [8, 197, 2304]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_179: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_178, [1576, 2304]);  view_178 = None
    mm_16: "f32[1576, 768]" = torch.ops.aten.mm.default(view_179, permute_124);  permute_124 = None
    permute_125: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_179, [1, 0])
    mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_125, view_121);  permute_125 = view_121 = None
    permute_126: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_31: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
    view_180: "f32[2304]" = torch.ops.aten.reshape.default(sum_31, [2304]);  sum_31 = None
    permute_127: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
    view_181: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_16, [8, 197, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_18: "f32[768]" = torch.ops.aten.slice.Tensor(view_180, 0, 0, 768)
    slice_20: "f32[768]" = torch.ops.aten.slice.Tensor(view_180, 0, 1536, 2304);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, primals_103);  primals_103 = None
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_161, 768)
    sum_32: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True)
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_161, mul_90);  mul_161 = None
    sum_33: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True);  mul_163 = None
    mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_33);  sum_33 = None
    sub_38: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_162, sum_32);  mul_162 = sum_32 = None
    sub_39: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_164);  sub_38 = mul_164 = None
    mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_39);  div_5 = sub_39 = None
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, mul_90);  mul_90 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 1]);  mul_166 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_181, [0, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_92, mul_165);  add_92 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_93, primals_99);  primals_99 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_93, view_120);  view_120 = None
    sum_36: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1], True);  mul_168 = None
    view_182: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_167, [1576, 768]);  mul_167 = None
    mm_18: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_183, permute_128);  permute_128 = None
    permute_129: "f32[768, 1576]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_129, view_119);  permute_129 = view_119 = None
    permute_130: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[768]" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
    permute_131: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_185: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_18, [8, 197, 3072]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_170: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
    mul_171: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_172: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_171, -0.5);  mul_171 = None
    exp_2: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_172);  mul_172 = None
    mul_173: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_174: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_173);  view_118 = mul_173 = None
    add_95: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_170, mul_174);  mul_170 = mul_174 = None
    mul_175: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_185, add_95);  view_185 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_175, [1576, 3072]);  mul_175 = None
    mm_20: "f32[1576, 768]" = torch.ops.aten.mm.default(view_186, permute_132);  permute_132 = None
    permute_133: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_133, view_117);  permute_133 = view_117 = None
    permute_134: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_38: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[3072]" = torch.ops.aten.reshape.default(sum_38, [3072]);  sum_38 = None
    permute_135: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_188: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_20, [8, 197, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_177: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_188, primals_100);  primals_100 = None
    mul_178: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_177, 768)
    sum_39: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
    mul_179: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_177, mul_84);  mul_177 = None
    sum_40: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
    mul_180: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, sum_40);  sum_40 = None
    sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_178, sum_39);  mul_178 = sum_39 = None
    sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_180);  sub_41 = mul_180 = None
    mul_181: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_42);  div_6 = sub_42 = None
    mul_182: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_84);  mul_84 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_96: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_93, mul_181);  add_93 = mul_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_183: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_96, primals_92);  primals_92 = None
    mul_184: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_96, view_116);  view_116 = None
    sum_43: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True);  mul_184 = None
    view_189: "f32[768]" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_190: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_183, [1576, 768]);  mul_183 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_190, permute_136);  permute_136 = None
    permute_137: "f32[768, 1576]" = torch.ops.aten.permute.default(view_190, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_137, view_115);  permute_137 = view_115 = None
    permute_138: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_190, [0], True);  view_190 = None
    view_191: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
    permute_139: "f32[768, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_192: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 197, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_193: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_192, [8, 197, 12, 64]);  view_192 = None
    permute_140: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_140, getitem_101, getitem_102, getitem_103, expand_10, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, True]);  permute_140 = getitem_101 = getitem_102 = getitem_103 = expand_10 = alias_14 = getitem_105 = getitem_106 = getitem_107 = None
    getitem_142: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_143: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_144: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_2[2]
    getitem_145: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_2[3];  _scaled_dot_product_efficient_attention_backward_2 = None
    sum_45: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_145, [0], True);  getitem_145 = None
    slice_scatter_4: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_45, -1, 0, 197);  sum_45 = None
    constant_pad_nd_14: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_4, [0, -3]);  slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_14, 0);  constant_pad_nd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_141: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_194: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_141, [38809, 12]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_112], view_194, True);  view_112 = view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_15: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_142, getitem_143, getitem_144]);  getitem_142 = getitem_143 = getitem_144 = None
    view_195: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_15, [3, 8, 12, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_142: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_195, [1, 3, 0, 2, 4]);  view_195 = None
    clone_52: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_196: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_52, [8, 197, 2304]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_197: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_196, [1576, 2304]);  view_196 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_197, permute_143);  permute_143 = None
    permute_144: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_144, view_109);  permute_144 = view_109 = None
    permute_145: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_46: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[2304]" = torch.ops.aten.reshape.default(sum_46, [2304]);  sum_46 = None
    permute_146: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_199: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_24, [8, 197, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_21: "f32[768]" = torch.ops.aten.slice.Tensor(view_198, 0, 0, 768)
    slice_23: "f32[768]" = torch.ops.aten.slice.Tensor(view_198, 0, 1536, 2304);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_186: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_199, primals_93);  primals_93 = None
    mul_187: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_186, 768)
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
    mul_188: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_186, mul_81);  mul_186 = None
    sum_48: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
    mul_189: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_48);  sum_48 = None
    sub_44: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_187, sum_47);  mul_187 = sum_47 = None
    sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_189);  sub_44 = mul_189 = None
    mul_190: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_45);  div_7 = sub_45 = None
    mul_191: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_199, mul_81);  mul_81 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
    sum_50: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_97: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_96, mul_190);  add_96 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_192: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_97, primals_89);  primals_89 = None
    mul_193: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_97, view_108);  view_108 = None
    sum_51: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 1], True);  mul_193 = None
    view_200: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_201: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_192, [1576, 768]);  mul_192 = None
    mm_26: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_201, permute_147);  permute_147 = None
    permute_148: "f32[768, 1576]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_148, view_107);  permute_148 = view_107 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[768]" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_203: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_26, [8, 197, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_195: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_61, 0.5);  add_61 = None
    mul_196: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_197: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_196, -0.5);  mul_196 = None
    exp_3: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_197);  mul_197 = None
    mul_198: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_199: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, mul_198);  view_106 = mul_198 = None
    add_99: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_195, mul_199);  mul_195 = mul_199 = None
    mul_200: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_203, add_99);  view_203 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_204: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_200, [1576, 3072]);  mul_200 = None
    mm_28: "f32[1576, 768]" = torch.ops.aten.mm.default(view_204, permute_151);  permute_151 = None
    permute_152: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_152, view_105);  permute_152 = view_105 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[3072]" = torch.ops.aten.reshape.default(sum_53, [3072]);  sum_53 = None
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_206: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_28, [8, 197, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_202: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_206, primals_90);  primals_90 = None
    mul_203: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_202, 768)
    sum_54: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
    mul_204: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_75);  mul_202 = None
    sum_55: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
    mul_205: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_75, sum_55);  sum_55 = None
    sub_47: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_203, sum_54);  mul_203 = sum_54 = None
    sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_205);  sub_47 = mul_205 = None
    mul_206: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_48);  div_8 = sub_48 = None
    mul_207: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_206, mul_75);  mul_75 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_206, [0, 1]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_100: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_97, mul_206);  add_97 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, primals_82);  primals_82 = None
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, view_104);  view_104 = None
    sum_58: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_209, [0, 1], True);  mul_209 = None
    view_207: "f32[768]" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_208: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_208, [1576, 768]);  mul_208 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_208, permute_155);  permute_155 = None
    permute_156: "f32[768, 1576]" = torch.ops.aten.permute.default(view_208, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_156, view_103);  permute_156 = view_103 = None
    permute_157: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_208, [0], True);  view_208 = None
    view_209: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_210: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_30, [8, 197, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_211: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_210, [8, 197, 12, 64]);  view_210 = None
    permute_159: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_159, getitem_90, getitem_91, getitem_92, expand_9, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, True]);  permute_159 = getitem_90 = getitem_91 = getitem_92 = expand_9 = alias_15 = getitem_94 = getitem_95 = getitem_96 = None
    getitem_146: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_147: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_148: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_3[2]
    getitem_149: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_3[3];  _scaled_dot_product_efficient_attention_backward_3 = None
    sum_60: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0], True);  getitem_149 = None
    slice_scatter_5: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_60, -1, 0, 197);  sum_60 = None
    constant_pad_nd_15: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_5, [0, -3]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_15, 0);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_160: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_212: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_160, [38809, 12]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_100], view_212, True);  view_100 = view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_16: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
    view_213: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_16, [3, 8, 12, 197, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_161: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_213, [1, 3, 0, 2, 4]);  view_213 = None
    clone_53: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_214: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_53, [8, 197, 2304]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_215: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_214, [1576, 2304]);  view_214 = None
    mm_32: "f32[1576, 768]" = torch.ops.aten.mm.default(view_215, permute_162);  permute_162 = None
    permute_163: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_163, view_97);  permute_163 = view_97 = None
    permute_164: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_61: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[2304]" = torch.ops.aten.reshape.default(sum_61, [2304]);  sum_61 = None
    permute_165: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_217: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_32, [8, 197, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_24: "f32[768]" = torch.ops.aten.slice.Tensor(view_216, 0, 0, 768)
    slice_26: "f32[768]" = torch.ops.aten.slice.Tensor(view_216, 0, 1536, 2304);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_211: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, primals_83);  primals_83 = None
    mul_212: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_211, 768)
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
    mul_213: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_211, mul_72);  mul_211 = None
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
    mul_214: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_63);  sum_63 = None
    sub_50: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_212, sum_62);  mul_212 = sum_62 = None
    sub_51: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_214);  sub_50 = mul_214 = None
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_51);  div_9 = sub_51 = None
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_72);  mul_72 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_100, mul_215);  add_100 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, primals_79);  primals_79 = None
    mul_218: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, view_96);  view_96 = None
    sum_66: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1], True);  mul_218 = None
    view_218: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_217, [1576, 768]);  mul_217 = None
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_219, permute_166);  permute_166 = None
    permute_167: "f32[768, 1576]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_167, view_95);  permute_167 = view_95 = None
    permute_168: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
    permute_169: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_221: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_220: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
    mul_221: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, view_94)
    mul_222: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_4: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_224: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, mul_223);  view_94 = mul_223 = None
    add_103: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_221, add_103);  view_221 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_222: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_225, [1576, 3072]);  mul_225 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_222, permute_170);  permute_170 = None
    permute_171: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_171, view_93);  permute_171 = view_93 = None
    permute_172: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[3072]" = torch.ops.aten.reshape.default(sum_68, [3072]);  sum_68 = None
    permute_173: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_224: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_227: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, primals_80);  primals_80 = None
    mul_228: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_66);  mul_227 = None
    sum_70: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_70);  sum_70 = None
    sub_53: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_69);  mul_228 = sum_69 = None
    sub_54: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_230);  sub_53 = mul_230 = None
    mul_231: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_54);  div_10 = sub_54 = None
    mul_232: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, mul_66);  mul_66 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_104: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, mul_231);  add_101 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_233: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, primals_72);  primals_72 = None
    mul_234: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, view_92);  view_92 = None
    sum_73: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1], True);  mul_234 = None
    view_225: "f32[768]" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_226: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_233, [1576, 768]);  mul_233 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_226, permute_174);  permute_174 = None
    permute_175: "f32[768, 1576]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_175, view_91);  permute_175 = view_91 = None
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[768]" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
    permute_177: "f32[768, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_228: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_229: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_228, [8, 197, 12, 64]);  view_228 = None
    permute_178: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_178, getitem_79, getitem_80, getitem_81, expand_8, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, True]);  permute_178 = getitem_79 = getitem_80 = getitem_81 = expand_8 = alias_16 = getitem_83 = getitem_84 = getitem_85 = None
    getitem_150: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_151: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_152: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_4[2]
    getitem_153: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_4[3];  _scaled_dot_product_efficient_attention_backward_4 = None
    sum_75: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_153, [0], True);  getitem_153 = None
    slice_scatter_6: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_75, -1, 0, 197);  sum_75 = None
    constant_pad_nd_16: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_6, [0, -3]);  slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_16, 0);  constant_pad_nd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_179: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_230: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_179, [38809, 12]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_88], view_230, True);  view_88 = view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_17: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
    view_231: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_17, [3, 8, 12, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_180: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_231, [1, 3, 0, 2, 4]);  view_231 = None
    clone_54: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    view_232: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_54, [8, 197, 2304]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_233: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_232, [1576, 2304]);  view_232 = None
    mm_40: "f32[1576, 768]" = torch.ops.aten.mm.default(view_233, permute_181);  permute_181 = None
    permute_182: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_182, view_85);  permute_182 = view_85 = None
    permute_183: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_76: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[2304]" = torch.ops.aten.reshape.default(sum_76, [2304]);  sum_76 = None
    permute_184: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    view_235: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_40, [8, 197, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_27: "f32[768]" = torch.ops.aten.slice.Tensor(view_234, 0, 0, 768)
    slice_29: "f32[768]" = torch.ops.aten.slice.Tensor(view_234, 0, 1536, 2304);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_236: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_235, primals_73);  primals_73 = None
    mul_237: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
    sum_77: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
    mul_238: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_63);  mul_236 = None
    sum_78: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    mul_239: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, sum_78);  sum_78 = None
    sub_56: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_77);  mul_237 = sum_77 = None
    sub_57: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_239);  sub_56 = mul_239 = None
    mul_240: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_57);  div_11 = sub_57 = None
    mul_241: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_235, mul_63);  mul_63 = None
    sum_79: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_104, mul_240);  add_104 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_242: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, primals_69);  primals_69 = None
    mul_243: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, view_84);  view_84 = None
    sum_81: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1], True);  mul_243 = None
    view_236: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_237: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_242, [1576, 768]);  mul_242 = None
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_237, permute_185);  permute_185 = None
    permute_186: "f32[768, 1576]" = torch.ops.aten.permute.default(view_237, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_186, view_83);  permute_186 = view_83 = None
    permute_187: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
    view_238: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
    permute_188: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
    view_239: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_245: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_47, 0.5);  add_47 = None
    mul_246: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, view_82)
    mul_247: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_246, -0.5);  mul_246 = None
    exp_5: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_247);  mul_247 = None
    mul_248: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_249: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, mul_248);  view_82 = mul_248 = None
    add_107: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_245, mul_249);  mul_245 = mul_249 = None
    mul_250: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_239, add_107);  view_239 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_240: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_250, [1576, 3072]);  mul_250 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_240, permute_189);  permute_189 = None
    permute_190: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_190, view_81);  permute_190 = view_81 = None
    permute_191: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
    view_241: "f32[3072]" = torch.ops.aten.reshape.default(sum_83, [3072]);  sum_83 = None
    permute_192: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    view_242: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_252: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_242, primals_70);  primals_70 = None
    mul_253: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_252, 768)
    sum_84: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
    mul_254: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_252, mul_57);  mul_252 = None
    sum_85: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
    mul_255: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_85);  sum_85 = None
    sub_59: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_253, sum_84);  mul_253 = sum_84 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_255);  sub_59 = mul_255 = None
    mul_256: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_60);  div_12 = sub_60 = None
    mul_257: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_242, mul_57);  mul_57 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_105, mul_256);  add_105 = mul_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_258: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, primals_62);  primals_62 = None
    mul_259: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, view_80);  view_80 = None
    sum_88: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1], True);  mul_259 = None
    view_243: "f32[768]" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_244: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_258, [1576, 768]);  mul_258 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_244, permute_193);  permute_193 = None
    permute_194: "f32[768, 1576]" = torch.ops.aten.permute.default(view_244, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_194, view_79);  permute_194 = view_79 = None
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True);  view_244 = None
    view_245: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    view_246: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_247: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_246, [8, 197, 12, 64]);  view_246 = None
    permute_197: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_197, getitem_68, getitem_69, getitem_70, expand_7, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, True]);  permute_197 = getitem_68 = getitem_69 = getitem_70 = expand_7 = alias_17 = getitem_72 = getitem_73 = getitem_74 = None
    getitem_154: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_155: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_156: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_5[2]
    getitem_157: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_5[3];  _scaled_dot_product_efficient_attention_backward_5 = None
    sum_90: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_157, [0], True);  getitem_157 = None
    slice_scatter_7: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_90, -1, 0, 197);  sum_90 = None
    constant_pad_nd_17: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_7, [0, -3]);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_17, 0);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_198: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_248: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_198, [38809, 12]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_76], view_248, True);  view_76 = view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_18: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
    view_249: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_18, [3, 8, 12, 197, 64]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_199: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_249, [1, 3, 0, 2, 4]);  view_249 = None
    clone_55: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_250: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_55, [8, 197, 2304]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_251: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_250, [1576, 2304]);  view_250 = None
    mm_48: "f32[1576, 768]" = torch.ops.aten.mm.default(view_251, permute_200);  permute_200 = None
    permute_201: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_251, [1, 0])
    mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_201, view_73);  permute_201 = view_73 = None
    permute_202: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_91: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
    view_252: "f32[2304]" = torch.ops.aten.reshape.default(sum_91, [2304]);  sum_91 = None
    permute_203: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_253: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_48, [8, 197, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_30: "f32[768]" = torch.ops.aten.slice.Tensor(view_252, 0, 0, 768)
    slice_32: "f32[768]" = torch.ops.aten.slice.Tensor(view_252, 0, 1536, 2304);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_261: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_253, primals_63);  primals_63 = None
    mul_262: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_261, 768)
    sum_92: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True)
    mul_263: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_261, mul_54);  mul_261 = None
    sum_93: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [2], True);  mul_263 = None
    mul_264: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_93);  sum_93 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_262, sum_92);  mul_262 = sum_92 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_264);  sub_62 = mul_264 = None
    mul_265: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_63);  div_13 = sub_63 = None
    mul_266: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_253, mul_54);  mul_54 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1]);  mul_266 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_253, [0, 1]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_108, mul_265);  add_108 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_267: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_59);  primals_59 = None
    mul_268: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, view_72);  view_72 = None
    sum_96: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True);  mul_268 = None
    view_254: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_255: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_267, [1576, 768]);  mul_267 = None
    mm_50: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_255, permute_204);  permute_204 = None
    permute_205: "f32[768, 1576]" = torch.ops.aten.permute.default(view_255, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_205, view_71);  permute_205 = view_71 = None
    permute_206: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
    view_256: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_257: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_50, [8, 197, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_270: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_271: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, view_70)
    mul_272: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_271, -0.5);  mul_271 = None
    exp_6: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_272);  mul_272 = None
    mul_273: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_274: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, mul_273);  view_70 = mul_273 = None
    add_111: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_270, mul_274);  mul_270 = mul_274 = None
    mul_275: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_257, add_111);  view_257 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_258: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_275, [1576, 3072]);  mul_275 = None
    mm_52: "f32[1576, 768]" = torch.ops.aten.mm.default(view_258, permute_208);  permute_208 = None
    permute_209: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_209, view_69);  permute_209 = view_69 = None
    permute_210: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_98: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[3072]" = torch.ops.aten.reshape.default(sum_98, [3072]);  sum_98 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_260: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_52, [8, 197, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_277: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_260, primals_60);  primals_60 = None
    mul_278: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_277, 768)
    sum_99: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
    mul_279: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_277, mul_48);  mul_277 = None
    sum_100: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    mul_280: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_48, sum_100);  sum_100 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_278, sum_99);  mul_278 = sum_99 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_280);  sub_65 = mul_280 = None
    mul_281: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_66);  div_14 = sub_66 = None
    mul_282: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_260, mul_48);  mul_48 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_109, mul_281);  add_109 = mul_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_283: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, primals_52);  primals_52 = None
    mul_284: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, view_68);  view_68 = None
    sum_103: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 1], True);  mul_284 = None
    view_261: "f32[768]" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_262: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_283, [1576, 768]);  mul_283 = None
    mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_262, permute_212);  permute_212 = None
    permute_213: "f32[768, 1576]" = torch.ops.aten.permute.default(view_262, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_213, view_67);  permute_213 = view_67 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True);  view_262 = None
    view_263: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
    permute_215: "f32[768, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_264: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 197, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_265: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_264, [8, 197, 12, 64]);  view_264 = None
    permute_216: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_216, getitem_57, getitem_58, getitem_59, expand_6, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, True]);  permute_216 = getitem_57 = getitem_58 = getitem_59 = expand_6 = alias_18 = getitem_61 = getitem_62 = getitem_63 = None
    getitem_158: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_159: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_160: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_6[2]
    getitem_161: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_6[3];  _scaled_dot_product_efficient_attention_backward_6 = None
    sum_105: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_161, [0], True);  getitem_161 = None
    slice_scatter_8: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_105, -1, 0, 197);  sum_105 = None
    constant_pad_nd_18: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_8, [0, -3]);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_18, 0);  constant_pad_nd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_217: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_266: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_217, [38809, 12]);  permute_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_64], view_266, True);  view_64 = view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_19: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
    view_267: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_19, [3, 8, 12, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_218: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_267, [1, 3, 0, 2, 4]);  view_267 = None
    clone_56: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_268: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_56, [8, 197, 2304]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_269: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_268, [1576, 2304]);  view_268 = None
    mm_56: "f32[1576, 768]" = torch.ops.aten.mm.default(view_269, permute_219);  permute_219 = None
    permute_220: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_220, view_61);  permute_220 = view_61 = None
    permute_221: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_106: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[2304]" = torch.ops.aten.reshape.default(sum_106, [2304]);  sum_106 = None
    permute_222: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    view_271: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_56, [8, 197, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_33: "f32[768]" = torch.ops.aten.slice.Tensor(view_270, 0, 0, 768)
    slice_35: "f32[768]" = torch.ops.aten.slice.Tensor(view_270, 0, 1536, 2304);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_286: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, primals_53);  primals_53 = None
    mul_287: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_286, 768)
    sum_107: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_45);  mul_286 = None
    sum_108: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_108);  sum_108 = None
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_287, sum_107);  mul_287 = sum_107 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_289);  sub_68 = mul_289 = None
    mul_290: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_69);  div_15 = sub_69 = None
    mul_291: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_45);  mul_45 = None
    sum_109: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_110: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_112, mul_290);  add_112 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_292: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_49);  primals_49 = None
    mul_293: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, view_60);  view_60 = None
    sum_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1], True);  mul_293 = None
    view_272: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_292, [1576, 768]);  mul_292 = None
    mm_58: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_273, permute_223);  permute_223 = None
    permute_224: "f32[768, 1576]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_224, view_59);  permute_224 = view_59 = None
    permute_225: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
    permute_226: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_275: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_58, [8, 197, 3072]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_295: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_33, 0.5);  add_33 = None
    mul_296: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_297: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_296, -0.5);  mul_296 = None
    exp_7: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_297);  mul_297 = None
    mul_298: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_299: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_298);  view_58 = mul_298 = None
    add_115: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_295, mul_299);  mul_295 = mul_299 = None
    mul_300: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_275, add_115);  view_275 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_300, [1576, 3072]);  mul_300 = None
    mm_60: "f32[1576, 768]" = torch.ops.aten.mm.default(view_276, permute_227);  permute_227 = None
    permute_228: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_228, view_57);  permute_228 = view_57 = None
    permute_229: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_113: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[3072]" = torch.ops.aten.reshape.default(sum_113, [3072]);  sum_113 = None
    permute_230: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_278: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_60, [8, 197, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_302: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_278, primals_50);  primals_50 = None
    mul_303: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_302, 768)
    sum_114: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
    mul_304: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_302, mul_39);  mul_302 = None
    sum_115: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
    mul_305: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_39, sum_115);  sum_115 = None
    sub_71: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_303, sum_114);  mul_303 = sum_114 = None
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_305);  sub_71 = mul_305 = None
    mul_306: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_72);  div_16 = sub_72 = None
    mul_307: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_278, mul_39);  mul_39 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_113, mul_306);  add_113 = mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_308: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_42);  primals_42 = None
    mul_309: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, view_56);  view_56 = None
    sum_118: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1], True);  mul_309 = None
    view_279: "f32[768]" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_280: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_308, [1576, 768]);  mul_308 = None
    mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_280, permute_231);  permute_231 = None
    permute_232: "f32[768, 1576]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_232, view_55);  permute_232 = view_55 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_282: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_62, [8, 197, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_283: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_282, [8, 197, 12, 64]);  view_282 = None
    permute_235: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_235, getitem_46, getitem_47, getitem_48, expand_5, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, True]);  permute_235 = getitem_46 = getitem_47 = getitem_48 = expand_5 = alias_19 = getitem_50 = getitem_51 = getitem_52 = None
    getitem_162: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_163: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_164: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_7[2]
    getitem_165: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_7[3];  _scaled_dot_product_efficient_attention_backward_7 = None
    sum_120: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_165, [0], True);  getitem_165 = None
    slice_scatter_9: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_120, -1, 0, 197);  sum_120 = None
    constant_pad_nd_19: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_9, [0, -3]);  slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_19, 0);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_236: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_284: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_236, [38809, 12]);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_52], view_284, True);  view_52 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_20: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
    view_285: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_20, [3, 8, 12, 197, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_237: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_285, [1, 3, 0, 2, 4]);  view_285 = None
    clone_57: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
    view_286: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_57, [8, 197, 2304]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_287: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_286, [1576, 2304]);  view_286 = None
    mm_64: "f32[1576, 768]" = torch.ops.aten.mm.default(view_287, permute_238);  permute_238 = None
    permute_239: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_239, view_49);  permute_239 = view_49 = None
    permute_240: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_121: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[2304]" = torch.ops.aten.reshape.default(sum_121, [2304]);  sum_121 = None
    permute_241: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_289: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_64, [8, 197, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_36: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 0, 768)
    slice_38: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 1536, 2304);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_311: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_43);  primals_43 = None
    mul_312: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_122: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_36);  mul_311 = None
    sum_123: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_123);  sum_123 = None
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_122);  mul_312 = sum_122 = None
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_314);  sub_74 = mul_314 = None
    mul_315: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_75);  div_17 = sub_75 = None
    mul_316: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_36);  mul_36 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_116, mul_315);  add_116 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_317: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_39);  primals_39 = None
    mul_318: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, view_48);  view_48 = None
    sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1], True);  mul_318 = None
    view_290: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_317, [1576, 768]);  mul_317 = None
    mm_66: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_291, permute_242);  permute_242 = None
    permute_243: "f32[768, 1576]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_243, view_47);  permute_243 = view_47 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    permute_245: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_293: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_66, [8, 197, 3072]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_320: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
    mul_321: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, view_46)
    mul_322: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_321, -0.5);  mul_321 = None
    exp_8: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_322);  mul_322 = None
    mul_323: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_324: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, mul_323);  view_46 = mul_323 = None
    add_119: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_320, mul_324);  mul_320 = mul_324 = None
    mul_325: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_293, add_119);  view_293 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_325, [1576, 3072]);  mul_325 = None
    mm_68: "f32[1576, 768]" = torch.ops.aten.mm.default(view_294, permute_246);  permute_246 = None
    permute_247: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_247, view_45);  permute_247 = view_45 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_128: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[3072]" = torch.ops.aten.reshape.default(sum_128, [3072]);  sum_128 = None
    permute_249: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_296: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_68, [8, 197, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_327: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, primals_40);  primals_40 = None
    mul_328: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_327, 768)
    sum_129: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
    mul_329: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_327, mul_30);  mul_327 = None
    sum_130: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
    mul_330: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_130);  sum_130 = None
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_328, sum_129);  mul_328 = sum_129 = None
    sub_78: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_330);  sub_77 = mul_330 = None
    mul_331: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_78);  div_18 = sub_78 = None
    mul_332: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, mul_30);  mul_30 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_296, [0, 1]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_117, mul_331);  add_117 = mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_333: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_32);  primals_32 = None
    mul_334: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, view_44);  view_44 = None
    sum_133: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1], True);  mul_334 = None
    view_297: "f32[768]" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_298: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_333, [1576, 768]);  mul_333 = None
    mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_298, permute_250);  permute_250 = None
    permute_251: "f32[768, 1576]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_43);  permute_251 = view_43 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_300: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_70, [8, 197, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_301: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_300, [8, 197, 12, 64]);  view_300 = None
    permute_254: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_254, getitem_35, getitem_36, getitem_37, expand_4, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, True]);  permute_254 = getitem_35 = getitem_36 = getitem_37 = expand_4 = alias_20 = getitem_39 = getitem_40 = getitem_41 = None
    getitem_166: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_167: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_168: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_8[2]
    getitem_169: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_8[3];  _scaled_dot_product_efficient_attention_backward_8 = None
    sum_135: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_169, [0], True);  getitem_169 = None
    slice_scatter_10: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_135, -1, 0, 197);  sum_135 = None
    constant_pad_nd_20: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_10, [0, -3]);  slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_20, 0);  constant_pad_nd_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_255: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_302: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_255, [38809, 12]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_40], view_302, True);  view_40 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_21: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
    view_303: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_21, [3, 8, 12, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_256: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_303, [1, 3, 0, 2, 4]);  view_303 = None
    clone_58: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    view_304: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_58, [8, 197, 2304]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_305: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_304, [1576, 2304]);  view_304 = None
    mm_72: "f32[1576, 768]" = torch.ops.aten.mm.default(view_305, permute_257);  permute_257 = None
    permute_258: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_258, view_37);  permute_258 = view_37 = None
    permute_259: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_136: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[2304]" = torch.ops.aten.reshape.default(sum_136, [2304]);  sum_136 = None
    permute_260: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    view_307: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_72, [8, 197, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_39: "f32[768]" = torch.ops.aten.slice.Tensor(view_306, 0, 0, 768)
    slice_41: "f32[768]" = torch.ops.aten.slice.Tensor(view_306, 0, 1536, 2304);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_336: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_307, primals_33);  primals_33 = None
    mul_337: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_336, 768)
    sum_137: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True)
    mul_338: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_336, mul_27);  mul_336 = None
    sum_138: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True);  mul_338 = None
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, sum_138);  sum_138 = None
    sub_80: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_337, sum_137);  mul_337 = sum_137 = None
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_339);  sub_80 = mul_339 = None
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_81);  div_19 = sub_81 = None
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_307, mul_27);  mul_27 = None
    sum_139: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1]);  mul_341 = None
    sum_140: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_121: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_120, mul_340);  add_120 = mul_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_29);  primals_29 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, view_36);  view_36 = None
    sum_141: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1], True);  mul_343 = None
    view_308: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_309: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_342, [1576, 768]);  mul_342 = None
    mm_74: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_309, permute_261);  permute_261 = None
    permute_262: "f32[768, 1576]" = torch.ops.aten.permute.default(view_309, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_262, view_35);  permute_262 = view_35 = None
    permute_263: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
    view_310: "f32[768]" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
    permute_264: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_311: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_74, [8, 197, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_345: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_19, 0.5);  add_19 = None
    mul_346: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, view_34)
    mul_347: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_346, -0.5);  mul_346 = None
    exp_9: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_347);  mul_347 = None
    mul_348: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_349: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, mul_348);  view_34 = mul_348 = None
    add_123: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_345, mul_349);  mul_345 = mul_349 = None
    mul_350: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_311, add_123);  view_311 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_312: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_350, [1576, 3072]);  mul_350 = None
    mm_76: "f32[1576, 768]" = torch.ops.aten.mm.default(view_312, permute_265);  permute_265 = None
    permute_266: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_266, view_33);  permute_266 = view_33 = None
    permute_267: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_143: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
    view_313: "f32[3072]" = torch.ops.aten.reshape.default(sum_143, [3072]);  sum_143 = None
    permute_268: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_314: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_76, [8, 197, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_352: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_314, primals_30);  primals_30 = None
    mul_353: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_352, 768)
    sum_144: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
    mul_354: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_352, mul_21);  mul_352 = None
    sum_145: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
    mul_355: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, sum_145);  sum_145 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_353, sum_144);  mul_353 = sum_144 = None
    sub_84: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_355);  sub_83 = mul_355 = None
    mul_356: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_84);  div_20 = sub_84 = None
    mul_357: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_314, mul_21);  mul_21 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_314, [0, 1]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_121, mul_356);  add_121 = mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_358: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, primals_22);  primals_22 = None
    mul_359: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, view_32);  view_32 = None
    sum_148: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1], True);  mul_359 = None
    view_315: "f32[768]" = torch.ops.aten.reshape.default(sum_148, [768]);  sum_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_316: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_358, [1576, 768]);  mul_358 = None
    mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_316, permute_269);  permute_269 = None
    permute_270: "f32[768, 1576]" = torch.ops.aten.permute.default(view_316, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_270, view_31);  permute_270 = view_31 = None
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
    view_317: "f32[768]" = torch.ops.aten.reshape.default(sum_149, [768]);  sum_149 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_318: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_78, [8, 197, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_319: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_318, [8, 197, 12, 64]);  view_318 = None
    permute_273: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_273, getitem_24, getitem_25, getitem_26, expand_3, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, True]);  permute_273 = getitem_24 = getitem_25 = getitem_26 = expand_3 = alias_21 = getitem_28 = getitem_29 = getitem_30 = None
    getitem_170: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
    getitem_171: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
    getitem_172: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_9[2]
    getitem_173: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_9[3];  _scaled_dot_product_efficient_attention_backward_9 = None
    sum_150: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0], True);  getitem_173 = None
    slice_scatter_11: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_150, -1, 0, 197);  sum_150 = None
    constant_pad_nd_21: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_11, [0, -3]);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_21, 0);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_274: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_320: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_274, [38809, 12]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_28], view_320, True);  view_28 = view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_22: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
    view_321: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_22, [3, 8, 12, 197, 64]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_275: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_321, [1, 3, 0, 2, 4]);  view_321 = None
    clone_59: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    view_322: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_59, [8, 197, 2304]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_323: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_322, [1576, 2304]);  view_322 = None
    mm_80: "f32[1576, 768]" = torch.ops.aten.mm.default(view_323, permute_276);  permute_276 = None
    permute_277: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_277, view_25);  permute_277 = view_25 = None
    permute_278: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_151: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[2304]" = torch.ops.aten.reshape.default(sum_151, [2304]);  sum_151 = None
    permute_279: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_325: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_80, [8, 197, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_42: "f32[768]" = torch.ops.aten.slice.Tensor(view_324, 0, 0, 768)
    slice_44: "f32[768]" = torch.ops.aten.slice.Tensor(view_324, 0, 1536, 2304);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_361: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_325, primals_23);  primals_23 = None
    mul_362: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_361, 768)
    sum_152: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2], True)
    mul_363: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_361, mul_18);  mul_361 = None
    sum_153: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True);  mul_363 = None
    mul_364: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_153);  sum_153 = None
    sub_86: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_362, sum_152);  mul_362 = sum_152 = None
    sub_87: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_364);  sub_86 = mul_364 = None
    mul_365: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_87);  div_21 = sub_87 = None
    mul_366: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_325, mul_18);  mul_18 = None
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1]);  mul_366 = None
    sum_155: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_325, [0, 1]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_124, mul_365);  add_124 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_367: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_19);  primals_19 = None
    mul_368: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, view_24);  view_24 = None
    sum_156: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 1], True);  mul_368 = None
    view_326: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_327: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_367, [1576, 768]);  mul_367 = None
    mm_82: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_327, permute_280);  permute_280 = None
    permute_281: "f32[768, 1576]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_281, view_23);  permute_281 = view_23 = None
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[768]" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
    permute_283: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_329: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_82, [8, 197, 3072]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_370: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_12, 0.5);  add_12 = None
    mul_371: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, view_22)
    mul_372: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_371, -0.5);  mul_371 = None
    exp_10: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_372);  mul_372 = None
    mul_373: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_374: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, mul_373);  view_22 = mul_373 = None
    add_127: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_370, mul_374);  mul_370 = mul_374 = None
    mul_375: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_329, add_127);  view_329 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_330: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_375, [1576, 3072]);  mul_375 = None
    mm_84: "f32[1576, 768]" = torch.ops.aten.mm.default(view_330, permute_284);  permute_284 = None
    permute_285: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_285, view_21);  permute_285 = view_21 = None
    permute_286: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_158: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[3072]" = torch.ops.aten.reshape.default(sum_158, [3072]);  sum_158 = None
    permute_287: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_332: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_84, [8, 197, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_377: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_332, primals_20);  primals_20 = None
    mul_378: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_377, 768)
    sum_159: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
    mul_379: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_377, mul_12);  mul_377 = None
    sum_160: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
    mul_380: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_12, sum_160);  sum_160 = None
    sub_89: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_378, sum_159);  mul_378 = sum_159 = None
    sub_90: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_380);  sub_89 = mul_380 = None
    mul_381: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_90);  div_22 = sub_90 = None
    mul_382: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_332, mul_12);  mul_12 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_332, [0, 1]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_128: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, mul_381);  add_125 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_383: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_12);  primals_12 = None
    mul_384: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, view_20);  view_20 = None
    sum_163: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1], True);  mul_384 = None
    view_333: "f32[768]" = torch.ops.aten.reshape.default(sum_163, [768]);  sum_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_334: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_383, [1576, 768]);  mul_383 = None
    mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_334, permute_288);  permute_288 = None
    permute_289: "f32[768, 1576]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_289, view_19);  permute_289 = view_19 = None
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[768]" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_336: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_86, [8, 197, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_337: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_336, [8, 197, 12, 64]);  view_336 = None
    permute_292: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_292, getitem_13, getitem_14, getitem_15, expand_2, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, True]);  permute_292 = getitem_13 = getitem_14 = getitem_15 = expand_2 = alias_22 = getitem_17 = getitem_18 = getitem_19 = None
    getitem_174: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
    getitem_175: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
    getitem_176: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_10[2]
    getitem_177: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_10[3];  _scaled_dot_product_efficient_attention_backward_10 = None
    sum_165: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_177, [0], True);  getitem_177 = None
    slice_scatter_12: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_165, -1, 0, 197);  sum_165 = None
    constant_pad_nd_22: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_12, [0, -3]);  slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_22, 0);  constant_pad_nd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_293: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_338: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_293, [38809, 12]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_3, [view_16], view_338, True);  view_16 = view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_23: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
    view_339: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_23, [3, 8, 12, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_294: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_339, [1, 3, 0, 2, 4]);  view_339 = None
    clone_60: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_340: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_60, [8, 197, 2304]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_341: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_340, [1576, 2304]);  view_340 = None
    mm_88: "f32[1576, 768]" = torch.ops.aten.mm.default(view_341, permute_295);  permute_295 = None
    permute_296: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_296, view_13);  permute_296 = view_13 = None
    permute_297: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_166: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[2304]" = torch.ops.aten.reshape.default(sum_166, [2304]);  sum_166 = None
    permute_298: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_343: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_88, [8, 197, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_45: "f32[768]" = torch.ops.aten.slice.Tensor(view_342, 0, 0, 768)
    slice_47: "f32[768]" = torch.ops.aten.slice.Tensor(view_342, 0, 1536, 2304);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_386: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_343, primals_13);  primals_13 = None
    mul_387: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_386, 768)
    sum_167: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
    mul_388: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_9);  mul_386 = None
    sum_168: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
    mul_389: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_168);  sum_168 = None
    sub_92: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_387, sum_167);  mul_387 = sum_167 = None
    sub_93: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_389);  sub_92 = mul_389 = None
    mul_390: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_93);  div_23 = sub_93 = None
    mul_391: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_343, mul_9);  mul_9 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
    sum_170: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_343, [0, 1]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_128, mul_390);  add_128 = mul_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_392: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_9);  primals_9 = None
    mul_393: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, view_12);  view_12 = None
    sum_171: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1], True);  mul_393 = None
    view_344: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_345: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_392, [1576, 768]);  mul_392 = None
    mm_90: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_345, permute_299);  permute_299 = None
    permute_300: "f32[768, 1576]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_300, view_11);  permute_300 = view_11 = None
    permute_301: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
    permute_302: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_347: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_90, [8, 197, 3072]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_395: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_5, 0.5);  add_5 = None
    mul_396: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, view_10)
    mul_397: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
    exp_11: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
    mul_398: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_399: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, mul_398);  view_10 = mul_398 = None
    add_131: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
    mul_400: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_347, add_131);  view_347 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_348: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_400, [1576, 3072]);  mul_400 = None
    mm_92: "f32[1576, 768]" = torch.ops.aten.mm.default(view_348, permute_303);  permute_303 = None
    permute_304: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_304, view_9);  permute_304 = view_9 = None
    permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_173: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
    view_349: "f32[3072]" = torch.ops.aten.reshape.default(sum_173, [3072]);  sum_173 = None
    permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_350: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_92, [8, 197, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_402: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_350, primals_10);  primals_10 = None
    mul_403: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_402, 768)
    sum_174: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
    mul_404: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_3);  mul_402 = None
    sum_175: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
    mul_405: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_175);  sum_175 = None
    sub_95: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_403, sum_174);  mul_403 = sum_174 = None
    sub_96: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_405);  sub_95 = mul_405 = None
    mul_406: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_96);  div_24 = sub_96 = None
    mul_407: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_350, mul_3);  mul_3 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_350, [0, 1]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_406);  add_129 = mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_408: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_2);  primals_2 = None
    mul_409: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, view_8);  view_8 = None
    sum_178: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 1], True);  mul_409 = None
    view_351: "f32[768]" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_352: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_408, [1576, 768]);  mul_408 = None
    mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_352, permute_307);  permute_307 = None
    permute_308: "f32[768, 1576]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_308, view_7);  permute_308 = view_7 = None
    permute_309: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[768]" = torch.ops.aten.reshape.default(sum_179, [768]);  sum_179 = None
    permute_310: "f32[768, 768]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_354: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_94, [8, 197, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_355: "f32[8, 197, 12, 64]" = torch.ops.aten.reshape.default(view_354, [8, 197, 12, 64]);  view_354 = None
    permute_311: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_355, [0, 2, 1, 3]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_311, getitem_2, getitem_3, getitem_4, expand_1, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, True]);  permute_311 = getitem_2 = getitem_3 = getitem_4 = expand_1 = alias_23 = getitem_6 = getitem_7 = getitem_8 = None
    getitem_178: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
    getitem_179: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
    getitem_180: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_backward_11[2]
    getitem_181: "f32[8, 12, 197, 197]" = _scaled_dot_product_efficient_attention_backward_11[3];  _scaled_dot_product_efficient_attention_backward_11 = None
    sum_180: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(getitem_181, [0], True);  getitem_181 = None
    slice_scatter_13: "f32[1, 12, 197, 200]" = torch.ops.aten.slice_scatter.default(full_default_2, sum_180, -1, 0, 197);  full_default_2 = sum_180 = None
    constant_pad_nd_23: "f32[1, 12, 197, 197]" = torch.ops.aten.constant_pad_nd.default(slice_scatter_13, [0, -3]);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(constant_pad_nd_23, 0);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_312: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_356: "f32[38809, 12]" = torch.ops.aten.reshape.default(permute_312, [38809, 12]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_11: "f32[732, 12]" = torch.ops.aten.index_put_.default(full_default_3, [view_4], view_356, True);  full_default_3 = view_4 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_24: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
    view_357: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.reshape.default(cat_24, [3, 8, 12, 197, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_313: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_357, [1, 3, 0, 2, 4]);  view_357 = None
    clone_61: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
    view_358: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_61, [8, 197, 2304]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_359: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_358, [1576, 2304]);  view_358 = None
    mm_96: "f32[1576, 768]" = torch.ops.aten.mm.default(view_359, permute_314);  permute_314 = None
    permute_315: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_315, view_1);  permute_315 = view_1 = None
    permute_316: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_181: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[2304]" = torch.ops.aten.reshape.default(sum_181, [2304]);  sum_181 = None
    permute_317: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_361: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_96, [8, 197, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_48: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 0, 768)
    slice_50: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 1536, 2304);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_411: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, primals_3);  primals_3 = None
    mul_412: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_411, 768)
    sum_182: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_411, mul);  mul_411 = None
    sum_183: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, sum_183);  sum_183 = None
    sub_98: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_412, sum_182);  mul_412 = sum_182 = None
    sub_99: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_414);  sub_98 = mul_414 = None
    div_25: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_415: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_99);  div_25 = sub_99 = None
    mul_416: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, mul);  mul = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_133: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_132, mul_415);  add_132 = mul_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    slice_51: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_133, 1, 0, 1)
    slice_52: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_133, 1, 1, 197);  add_133 = None
    sum_186: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_51, [0], True);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_318: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_52, [0, 2, 1]);  slice_52 = None
    view_362: "f32[8, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_318, [8, 768, 14, 14]);  permute_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_362, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_362, primals_224, primals_124, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_362 = primals_224 = primals_124 = None
    getitem_183: "f32[768, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return [sum_186, view_351, sum_184, sum_185, slice_48, slice_50, permute_317, index_put_11, view_344, sum_176, sum_177, view_333, sum_169, sum_170, slice_45, slice_47, permute_298, index_put_10, view_326, sum_161, sum_162, view_315, sum_154, sum_155, slice_42, slice_44, permute_279, index_put_9, view_308, sum_146, sum_147, view_297, sum_139, sum_140, slice_39, slice_41, permute_260, index_put_8, view_290, sum_131, sum_132, view_279, sum_124, sum_125, slice_36, slice_38, permute_241, index_put_7, view_272, sum_116, sum_117, view_261, sum_109, sum_110, slice_33, slice_35, permute_222, index_put_6, view_254, sum_101, sum_102, view_243, sum_94, sum_95, slice_30, slice_32, permute_203, index_put_5, view_236, sum_86, sum_87, view_225, sum_79, sum_80, slice_27, slice_29, permute_184, index_put_4, view_218, sum_71, sum_72, view_207, sum_64, sum_65, slice_24, slice_26, permute_165, index_put_3, view_200, sum_56, sum_57, view_189, sum_49, sum_50, slice_21, slice_23, permute_146, index_put_2, view_182, sum_41, sum_42, view_171, sum_34, sum_35, slice_18, slice_20, permute_127, index_put_1, view_164, sum_26, sum_27, view_153, sum_19, sum_20, slice_15, slice_17, permute_108, index_put, view_146, sum_11, sum_12, sum_4, sum_5, getitem_183, sum_187, permute_310, view_353, permute_306, view_349, permute_302, view_346, permute_291, view_335, permute_287, view_331, permute_283, view_328, permute_272, view_317, permute_268, view_313, permute_264, view_310, permute_253, view_299, permute_249, view_295, permute_245, view_292, permute_234, view_281, permute_230, view_277, permute_226, view_274, permute_215, view_263, permute_211, view_259, permute_207, view_256, permute_196, view_245, permute_192, view_241, permute_188, view_238, permute_177, view_227, permute_173, view_223, permute_169, view_220, permute_158, view_209, permute_154, view_205, permute_150, view_202, permute_139, view_191, permute_135, view_187, permute_131, view_184, permute_120, view_173, permute_116, view_169, permute_112, view_166, permute_101, view_155, permute_97, view_151, permute_93, view_148, permute_89, view_145, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    