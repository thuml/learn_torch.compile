from __future__ import annotations



def forward(self, primals_2: "f32[768]", primals_3: "f32[768]", primals_9: "f32[768]", primals_10: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_39: "f32[768]", primals_40: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_49: "f32[768]", primals_50: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_69: "f32[768]", primals_70: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_79: "f32[768]", primals_80: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_99: "f32[768]", primals_100: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_109: "f32[768]", primals_110: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_122: "f32[768]", primals_124: "f32[768, 3, 16, 16]", primals_224: "f32[8, 3, 224, 224]", cat: "f32[8, 197, 768]", getitem_1: "f32[8, 197, 1]", rsqrt: "f32[8, 197, 1]", view_1: "f32[1576, 768]", view_4: "i64[38809]", view_13: "f32[1576, 768]", addmm_1: "f32[1576, 768]", mul_5: "f32[8, 197, 768]", view_15: "f32[1576, 768]", addmm_2: "f32[1576, 3072]", view_17: "f32[1576, 3072]", addmm_3: "f32[1576, 768]", mul_11: "f32[8, 197, 768]", view_19: "f32[1576, 768]", view_22: "i64[38809]", view_31: "f32[1576, 768]", addmm_5: "f32[1576, 768]", mul_16: "f32[8, 197, 768]", view_33: "f32[1576, 768]", addmm_6: "f32[1576, 3072]", view_35: "f32[1576, 3072]", addmm_7: "f32[1576, 768]", mul_22: "f32[8, 197, 768]", view_37: "f32[1576, 768]", view_40: "i64[38809]", view_49: "f32[1576, 768]", addmm_9: "f32[1576, 768]", mul_27: "f32[8, 197, 768]", view_51: "f32[1576, 768]", addmm_10: "f32[1576, 3072]", view_53: "f32[1576, 3072]", addmm_11: "f32[1576, 768]", mul_33: "f32[8, 197, 768]", view_55: "f32[1576, 768]", view_58: "i64[38809]", view_67: "f32[1576, 768]", addmm_13: "f32[1576, 768]", mul_38: "f32[8, 197, 768]", view_69: "f32[1576, 768]", addmm_14: "f32[1576, 3072]", view_71: "f32[1576, 3072]", addmm_15: "f32[1576, 768]", mul_44: "f32[8, 197, 768]", view_73: "f32[1576, 768]", view_76: "i64[38809]", view_85: "f32[1576, 768]", addmm_17: "f32[1576, 768]", mul_49: "f32[8, 197, 768]", view_87: "f32[1576, 768]", addmm_18: "f32[1576, 3072]", view_89: "f32[1576, 3072]", addmm_19: "f32[1576, 768]", mul_55: "f32[8, 197, 768]", view_91: "f32[1576, 768]", view_94: "i64[38809]", view_103: "f32[1576, 768]", addmm_21: "f32[1576, 768]", mul_60: "f32[8, 197, 768]", view_105: "f32[1576, 768]", addmm_22: "f32[1576, 3072]", view_107: "f32[1576, 3072]", addmm_23: "f32[1576, 768]", mul_66: "f32[8, 197, 768]", view_109: "f32[1576, 768]", view_112: "i64[38809]", view_121: "f32[1576, 768]", addmm_25: "f32[1576, 768]", mul_71: "f32[8, 197, 768]", view_123: "f32[1576, 768]", addmm_26: "f32[1576, 3072]", view_125: "f32[1576, 3072]", addmm_27: "f32[1576, 768]", mul_77: "f32[8, 197, 768]", view_127: "f32[1576, 768]", view_130: "i64[38809]", view_139: "f32[1576, 768]", addmm_29: "f32[1576, 768]", mul_82: "f32[8, 197, 768]", view_141: "f32[1576, 768]", addmm_30: "f32[1576, 3072]", view_143: "f32[1576, 3072]", addmm_31: "f32[1576, 768]", mul_88: "f32[8, 197, 768]", view_145: "f32[1576, 768]", view_148: "i64[38809]", view_157: "f32[1576, 768]", addmm_33: "f32[1576, 768]", mul_93: "f32[8, 197, 768]", view_159: "f32[1576, 768]", addmm_34: "f32[1576, 3072]", view_161: "f32[1576, 3072]", addmm_35: "f32[1576, 768]", mul_99: "f32[8, 197, 768]", view_163: "f32[1576, 768]", view_166: "i64[38809]", view_175: "f32[1576, 768]", addmm_37: "f32[1576, 768]", mul_104: "f32[8, 197, 768]", view_177: "f32[1576, 768]", addmm_38: "f32[1576, 3072]", view_179: "f32[1576, 3072]", addmm_39: "f32[1576, 768]", mul_110: "f32[8, 197, 768]", view_181: "f32[1576, 768]", view_184: "i64[38809]", view_193: "f32[1576, 768]", addmm_41: "f32[1576, 768]", mul_115: "f32[8, 197, 768]", view_195: "f32[1576, 768]", addmm_42: "f32[1576, 3072]", view_197: "f32[1576, 3072]", addmm_43: "f32[1576, 768]", mul_121: "f32[8, 197, 768]", view_199: "f32[1576, 768]", view_202: "i64[38809]", view_211: "f32[1576, 768]", addmm_45: "f32[1576, 768]", mul_126: "f32[8, 197, 768]", view_213: "f32[1576, 768]", addmm_46: "f32[1576, 3072]", view_215: "f32[1576, 3072]", addmm_47: "f32[1576, 768]", mul_132: "f32[8, 768]", clone_97: "f32[8, 768]", permute_98: "f32[1000, 768]", div_12: "f32[8, 1]", permute_102: "f32[768, 3072]", permute_106: "f32[3072, 768]", div_14: "f32[8, 197, 1]", permute_110: "f32[768, 768]", permute_115: "f32[96, 197, 197]", permute_116: "f32[96, 64, 197]", alias_12: "f32[8, 12, 197, 197]", permute_117: "f32[96, 64, 197]", permute_118: "f32[96, 197, 64]", permute_122: "f32[2304, 768]", div_15: "f32[8, 197, 1]", permute_126: "f32[768, 3072]", permute_130: "f32[3072, 768]", div_16: "f32[8, 197, 1]", permute_134: "f32[768, 768]", permute_139: "f32[96, 197, 197]", permute_140: "f32[96, 64, 197]", alias_13: "f32[8, 12, 197, 197]", permute_141: "f32[96, 64, 197]", permute_142: "f32[96, 197, 64]", permute_146: "f32[2304, 768]", div_17: "f32[8, 197, 1]", permute_150: "f32[768, 3072]", permute_154: "f32[3072, 768]", div_18: "f32[8, 197, 1]", permute_158: "f32[768, 768]", permute_163: "f32[96, 197, 197]", permute_164: "f32[96, 64, 197]", alias_14: "f32[8, 12, 197, 197]", permute_165: "f32[96, 64, 197]", permute_166: "f32[96, 197, 64]", permute_170: "f32[2304, 768]", div_19: "f32[8, 197, 1]", permute_174: "f32[768, 3072]", permute_178: "f32[3072, 768]", div_20: "f32[8, 197, 1]", permute_182: "f32[768, 768]", permute_187: "f32[96, 197, 197]", permute_188: "f32[96, 64, 197]", alias_15: "f32[8, 12, 197, 197]", permute_189: "f32[96, 64, 197]", permute_190: "f32[96, 197, 64]", permute_194: "f32[2304, 768]", div_21: "f32[8, 197, 1]", permute_198: "f32[768, 3072]", permute_202: "f32[3072, 768]", div_22: "f32[8, 197, 1]", permute_206: "f32[768, 768]", permute_211: "f32[96, 197, 197]", permute_212: "f32[96, 64, 197]", alias_16: "f32[8, 12, 197, 197]", permute_213: "f32[96, 64, 197]", permute_214: "f32[96, 197, 64]", permute_218: "f32[2304, 768]", div_23: "f32[8, 197, 1]", permute_222: "f32[768, 3072]", permute_226: "f32[3072, 768]", div_24: "f32[8, 197, 1]", permute_230: "f32[768, 768]", permute_235: "f32[96, 197, 197]", permute_236: "f32[96, 64, 197]", alias_17: "f32[8, 12, 197, 197]", permute_237: "f32[96, 64, 197]", permute_238: "f32[96, 197, 64]", permute_242: "f32[2304, 768]", div_25: "f32[8, 197, 1]", permute_246: "f32[768, 3072]", permute_250: "f32[3072, 768]", div_26: "f32[8, 197, 1]", permute_254: "f32[768, 768]", permute_259: "f32[96, 197, 197]", permute_260: "f32[96, 64, 197]", alias_18: "f32[8, 12, 197, 197]", permute_261: "f32[96, 64, 197]", permute_262: "f32[96, 197, 64]", permute_266: "f32[2304, 768]", div_27: "f32[8, 197, 1]", permute_270: "f32[768, 3072]", permute_274: "f32[3072, 768]", div_28: "f32[8, 197, 1]", permute_278: "f32[768, 768]", permute_283: "f32[96, 197, 197]", permute_284: "f32[96, 64, 197]", alias_19: "f32[8, 12, 197, 197]", permute_285: "f32[96, 64, 197]", permute_286: "f32[96, 197, 64]", permute_290: "f32[2304, 768]", div_29: "f32[8, 197, 1]", permute_294: "f32[768, 3072]", permute_298: "f32[3072, 768]", div_30: "f32[8, 197, 1]", permute_302: "f32[768, 768]", permute_307: "f32[96, 197, 197]", permute_308: "f32[96, 64, 197]", alias_20: "f32[8, 12, 197, 197]", permute_309: "f32[96, 64, 197]", permute_310: "f32[96, 197, 64]", permute_314: "f32[2304, 768]", div_31: "f32[8, 197, 1]", permute_318: "f32[768, 3072]", permute_322: "f32[3072, 768]", div_32: "f32[8, 197, 1]", permute_326: "f32[768, 768]", permute_331: "f32[96, 197, 197]", permute_332: "f32[96, 64, 197]", alias_21: "f32[8, 12, 197, 197]", permute_333: "f32[96, 64, 197]", permute_334: "f32[96, 197, 64]", permute_338: "f32[2304, 768]", div_33: "f32[8, 197, 1]", permute_342: "f32[768, 3072]", permute_346: "f32[3072, 768]", div_34: "f32[8, 197, 1]", permute_350: "f32[768, 768]", permute_355: "f32[96, 197, 197]", permute_356: "f32[96, 64, 197]", alias_22: "f32[8, 12, 197, 197]", permute_357: "f32[96, 64, 197]", permute_358: "f32[96, 197, 64]", permute_362: "f32[2304, 768]", div_35: "f32[8, 197, 1]", permute_366: "f32[768, 3072]", permute_370: "f32[3072, 768]", div_36: "f32[8, 197, 1]", permute_374: "f32[768, 768]", permute_379: "f32[96, 197, 197]", permute_380: "f32[96, 64, 197]", alias_23: "f32[8, 12, 197, 197]", permute_381: "f32[96, 64, 197]", permute_382: "f32[96, 197, 64]", permute_386: "f32[2304, 768]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:408, code: x = self.pos_drop(x)
    clone: "f32[8, 197, 768]" = torch.ops.aten.clone.default(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_14: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_14);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_16: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_18: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_32: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_14: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476)
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_14: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_50: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_52: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_30: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476)
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_22: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_68: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476)
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_72: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_86: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_38: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_86);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_38: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_46: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_63: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_122: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_54: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_122);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.7071067811865476)
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_126: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_140: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_62: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_85: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476)
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_144: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_158: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_70: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_160: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476)
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_70: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_162: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_176: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_78: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_178: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476)
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_78: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_180: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_194: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_86: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_194);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_196: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_86: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_198);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_212: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_94: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_212);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_94: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_216: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_98);  permute_98 = None
    permute_99: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_99, clone_97);  permute_99 = clone_97 = None
    permute_100: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_217: "f32[1000]" = torch.ops.aten.view.default(sum_13, [1000]);  sum_13 = None
    permute_101: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_135: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, primals_122);  primals_122 = None
    mul_136: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_135, 768)
    sum_14: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [1], True)
    mul_137: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_135, mul_132);  mul_135 = None
    sum_15: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [1], True);  mul_137 = None
    mul_138: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_132, sum_15);  sum_15 = None
    sub_38: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mul_136, sum_14);  mul_136 = sum_14 = None
    sub_39: "f32[8, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_138);  sub_38 = mul_138 = None
    mul_139: "f32[8, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_39);  div_12 = sub_39 = None
    mul_140: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, mul_132);  mul_132 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0]);  mul_140 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm, [0]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    unsqueeze_12: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_139, 1);  mul_139 = None
    expand_49: "f32[8, 196, 768]" = torch.ops.aten.expand.default(unsqueeze_12, [8, 196, 768]);  unsqueeze_12 = None
    div_13: "f32[8, 196, 768]" = torch.ops.aten.div.Scalar(expand_49, 196);  expand_49 = None
    full_default: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, div_13, 1, 1, 9223372036854775807);  div_13 = None
    slice_scatter_1: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter, 0, 0, 9223372036854775807);  full_default = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_141: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_1, primals_119);  primals_119 = None
    mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_1, clone_96);  clone_96 = None
    sum_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1], True);  mul_142 = None
    view_218: "f32[768]" = torch.ops.aten.view.default(sum_18, [768]);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[1576, 768]" = torch.ops.aten.view.default(mul_141, [1576, 768]);  mul_141 = None
    mm_2: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_219, permute_102);  permute_102 = None
    permute_103: "f32[768, 1576]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_103, view_215);  permute_103 = view_215 = None
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    permute_105: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    view_221: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_2, [8, 197, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_144: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
    mul_145: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, view_214)
    mul_146: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_145, -0.5);  mul_145 = None
    exp_12: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_146);  mul_146 = None
    mul_147: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_148: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, mul_147);  view_214 = mul_147 = None
    add_99: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_144, mul_148);  mul_144 = mul_148 = None
    mul_149: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_221, add_99);  view_221 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_222: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_149, [1576, 3072]);  mul_149 = None
    mm_4: "f32[1576, 768]" = torch.ops.aten.mm.default(view_222, permute_106);  permute_106 = None
    permute_107: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_107, view_213);  permute_107 = view_213 = None
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[3072]" = torch.ops.aten.view.default(sum_20, [3072]);  sum_20 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    view_224: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_4, [8, 197, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_151: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, primals_120);  primals_120 = None
    mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_151, 768)
    sum_21: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True)
    mul_153: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_126);  mul_151 = None
    sum_22: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_126, sum_22);  sum_22 = None
    sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_152, sum_21);  mul_152 = sum_21 = None
    sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_154);  sub_41 = mul_154 = None
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_42);  div_14 = sub_42 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, mul_126);  mul_126 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1]);  mul_156 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_100: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(slice_scatter_1, mul_155);  slice_scatter_1 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, primals_112);  primals_112 = None
    mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, clone_94);  clone_94 = None
    sum_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_225: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_226: "f32[1576, 768]" = torch.ops.aten.view.default(mul_157, [1576, 768]);  mul_157 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_226, permute_110);  permute_110 = None
    permute_111: "f32[768, 1576]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_111, view_211);  permute_111 = view_211 = None
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_228: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_6, [8, 197, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_229: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_228, [8, 197, 12, 64]);  view_228 = None
    permute_114: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_98: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_230: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_98, [96, 197, 64]);  clone_98 = None
    bmm_24: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_115, view_230);  permute_115 = None
    bmm_25: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_230, permute_116);  view_230 = permute_116 = None
    view_231: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_24, [8, 12, 197, 64]);  bmm_24 = None
    view_232: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_25, [8, 12, 197, 197]);  bmm_25 = None
    mul_159: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_232, alias_12);  view_232 = None
    sum_27: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [-1], True)
    mul_160: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_12, sum_27);  alias_12 = sum_27 = None
    sub_43: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    sum_28: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_43, [0], True)
    view_233: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_43, [96, 197, 197]);  sub_43 = None
    bmm_26: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_117, view_233);  permute_117 = None
    bmm_27: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_233, permute_118);  view_233 = permute_118 = None
    view_234: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_26, [8, 12, 64, 197]);  bmm_26 = None
    view_235: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_27, [8, 12, 197, 64]);  bmm_27 = None
    mul_161: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_234, 0.3535533905932738);  view_234 = None
    permute_119: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_161, [0, 1, 3, 2]);  mul_161 = None
    mul_162: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_235, 0.3535533905932738);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_28, 0);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_120: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_236: "f32[38809, 12]" = torch.ops.aten.view.default(permute_120, [38809, 12]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_default_2: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_202], view_236, True);  view_202 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_13: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_162, permute_119, view_231]);  mul_162 = permute_119 = view_231 = None
    view_237: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_13, [3, 8, 12, 197, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_121: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_237, [1, 3, 0, 2, 4]);  view_237 = None
    clone_99: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_238: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_99, [8, 197, 2304]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_239: "f32[1576, 2304]" = torch.ops.aten.view.default(view_238, [1576, 2304]);  view_238 = None
    mm_8: "f32[1576, 768]" = torch.ops.aten.mm.default(view_239, permute_122);  permute_122 = None
    permute_123: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_123, view_199);  permute_123 = view_199 = None
    permute_124: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_29: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[2304]" = torch.ops.aten.view.default(sum_29, [2304]);  sum_29 = None
    permute_125: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_241: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_8, [8, 197, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_3: "f32[768]" = torch.ops.aten.slice.Tensor(view_240, 0, 0, 768)
    slice_5: "f32[768]" = torch.ops.aten.slice.Tensor(view_240, 0, 1536, 2304);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, primals_113);  primals_113 = None
    mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_30: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_121);  mul_164 = None
    sum_31: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, sum_31);  sum_31 = None
    sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_30);  mul_165 = sum_30 = None
    sub_46: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_167);  sub_45 = mul_167 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, mul_121);  mul_121 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_100, mul_168);  add_100 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, primals_109);  primals_109 = None
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, clone_88);  clone_88 = None
    sum_34: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1], True);  mul_171 = None
    view_242: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_243: "f32[1576, 768]" = torch.ops.aten.view.default(mul_170, [1576, 768]);  mul_170 = None
    mm_10: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_243, permute_126);  permute_126 = None
    permute_127: "f32[768, 1576]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_127, view_197);  permute_127 = view_197 = None
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_129: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    view_245: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_10, [8, 197, 3072]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_173: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_174: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, view_196)
    mul_175: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_174, -0.5);  mul_174 = None
    exp_13: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_175);  mul_175 = None
    mul_176: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_177: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, mul_176);  view_196 = mul_176 = None
    add_103: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_173, mul_177);  mul_173 = mul_177 = None
    mul_178: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_245, add_103);  view_245 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_178, [1576, 3072]);  mul_178 = None
    mm_12: "f32[1576, 768]" = torch.ops.aten.mm.default(view_246, permute_130);  permute_130 = None
    permute_131: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_131, view_195);  permute_131 = view_195 = None
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[3072]" = torch.ops.aten.view.default(sum_36, [3072]);  sum_36 = None
    permute_133: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    view_248: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_12, [8, 197, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_180: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_248, primals_110);  primals_110 = None
    mul_181: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_180, 768)
    sum_37: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
    mul_182: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_180, mul_115);  mul_180 = None
    sum_38: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
    mul_183: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_115, sum_38);  sum_38 = None
    sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_181, sum_37);  mul_181 = sum_37 = None
    sub_49: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_183);  sub_48 = mul_183 = None
    mul_184: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
    mul_185: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_248, mul_115);  mul_115 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_104: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, mul_184);  add_101 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_186: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, primals_102);  primals_102 = None
    mul_187: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, clone_86);  clone_86 = None
    sum_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1], True);  mul_187 = None
    view_249: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_250: "f32[1576, 768]" = torch.ops.aten.view.default(mul_186, [1576, 768]);  mul_186 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_250, permute_134);  permute_134 = None
    permute_135: "f32[768, 1576]" = torch.ops.aten.permute.default(view_250, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_135, view_193);  permute_135 = view_193 = None
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
    view_251: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_252: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_14, [8, 197, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_253: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_252, [8, 197, 12, 64]);  view_252 = None
    permute_138: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_100: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
    view_254: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_100, [96, 197, 64]);  clone_100 = None
    bmm_28: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_139, view_254);  permute_139 = None
    bmm_29: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_254, permute_140);  view_254 = permute_140 = None
    view_255: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_28, [8, 12, 197, 64]);  bmm_28 = None
    view_256: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_29, [8, 12, 197, 197]);  bmm_29 = None
    mul_188: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_256, alias_13);  view_256 = None
    sum_43: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [-1], True)
    mul_189: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_13, sum_43);  alias_13 = sum_43 = None
    sub_50: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    sum_44: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_50, [0], True)
    view_257: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_50, [96, 197, 197]);  sub_50 = None
    bmm_30: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_141, view_257);  permute_141 = None
    bmm_31: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_257, permute_142);  view_257 = permute_142 = None
    view_258: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_30, [8, 12, 64, 197]);  bmm_30 = None
    view_259: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_31, [8, 12, 197, 64]);  bmm_31 = None
    mul_190: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_258, 0.3535533905932738);  view_258 = None
    permute_143: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_190, [0, 1, 3, 2]);  mul_190 = None
    mul_191: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_259, 0.3535533905932738);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_44, 0);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_144: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_260: "f32[38809, 12]" = torch.ops.aten.view.default(permute_144, [38809, 12]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_184], view_260, True);  view_184 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_14: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_191, permute_143, view_255]);  mul_191 = permute_143 = view_255 = None
    view_261: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_14, [3, 8, 12, 197, 64]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_145: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_261, [1, 3, 0, 2, 4]);  view_261 = None
    clone_101: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_101, [8, 197, 2304]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_263: "f32[1576, 2304]" = torch.ops.aten.view.default(view_262, [1576, 2304]);  view_262 = None
    mm_16: "f32[1576, 768]" = torch.ops.aten.mm.default(view_263, permute_146);  permute_146 = None
    permute_147: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_147, view_181);  permute_147 = view_181 = None
    permute_148: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_45: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[2304]" = torch.ops.aten.view.default(sum_45, [2304]);  sum_45 = None
    permute_149: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_265: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_16, [8, 197, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_6: "f32[768]" = torch.ops.aten.slice.Tensor(view_264, 0, 0, 768)
    slice_8: "f32[768]" = torch.ops.aten.slice.Tensor(view_264, 0, 1536, 2304);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_193: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_265, primals_103);  primals_103 = None
    mul_194: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_193, 768)
    sum_46: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
    mul_195: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_193, mul_110);  mul_193 = None
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    mul_196: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_110, sum_47);  sum_47 = None
    sub_52: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_194, sum_46);  mul_194 = sum_46 = None
    sub_53: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_196);  sub_52 = mul_196 = None
    mul_197: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_53);  div_17 = sub_53 = None
    mul_198: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_110);  mul_110 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_104, mul_197);  add_104 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_199: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, primals_99);  primals_99 = None
    mul_200: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, clone_80);  clone_80 = None
    sum_50: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1], True);  mul_200 = None
    view_266: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_267: "f32[1576, 768]" = torch.ops.aten.view.default(mul_199, [1576, 768]);  mul_199 = None
    mm_18: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_267, permute_150);  permute_150 = None
    permute_151: "f32[768, 1576]" = torch.ops.aten.permute.default(view_267, [1, 0])
    mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_179);  permute_151 = view_179 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True);  view_267 = None
    view_268: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_269: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_18, [8, 197, 3072]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_202: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_203: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, view_178)
    mul_204: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
    exp_14: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_204);  mul_204 = None
    mul_205: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_206: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, mul_205);  view_178 = mul_205 = None
    add_107: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
    mul_207: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_269, add_107);  view_269 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_270: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_207, [1576, 3072]);  mul_207 = None
    mm_20: "f32[1576, 768]" = torch.ops.aten.mm.default(view_270, permute_154);  permute_154 = None
    permute_155: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_177);  permute_155 = view_177 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_52: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[3072]" = torch.ops.aten.view.default(sum_52, [3072]);  sum_52 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_272: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_20, [8, 197, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_272, primals_100);  primals_100 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_209, 768)
    sum_53: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
    mul_211: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_209, mul_104);  mul_209 = None
    sum_54: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    mul_212: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_104, sum_54);  sum_54 = None
    sub_55: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_210, sum_53);  mul_210 = sum_53 = None
    sub_56: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_212);  sub_55 = mul_212 = None
    mul_213: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_56);  div_18 = sub_56 = None
    mul_214: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_272, mul_104);  mul_104 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_105, mul_213);  add_105 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, primals_92);  primals_92 = None
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, clone_78);  clone_78 = None
    sum_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1], True);  mul_216 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_274: "f32[1576, 768]" = torch.ops.aten.view.default(mul_215, [1576, 768]);  mul_215 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_274, permute_158);  permute_158 = None
    permute_159: "f32[768, 1576]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_159, view_175);  permute_159 = view_175 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_276: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_277: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_276, [8, 197, 12, 64]);  view_276 = None
    permute_162: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_102: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_278: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_102, [96, 197, 64]);  clone_102 = None
    bmm_32: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_163, view_278);  permute_163 = None
    bmm_33: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_278, permute_164);  view_278 = permute_164 = None
    view_279: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_32, [8, 12, 197, 64]);  bmm_32 = None
    view_280: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_33, [8, 12, 197, 197]);  bmm_33 = None
    mul_217: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_280, alias_14);  view_280 = None
    sum_59: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
    mul_218: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_14, sum_59);  alias_14 = sum_59 = None
    sub_57: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    sum_60: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_57, [0], True)
    view_281: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_57, [96, 197, 197]);  sub_57 = None
    bmm_34: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_165, view_281);  permute_165 = None
    bmm_35: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_281, permute_166);  view_281 = permute_166 = None
    view_282: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_34, [8, 12, 64, 197]);  bmm_34 = None
    view_283: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_35, [8, 12, 197, 64]);  bmm_35 = None
    mul_219: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_282, 0.3535533905932738);  view_282 = None
    permute_167: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_219, [0, 1, 3, 2]);  mul_219 = None
    mul_220: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_283, 0.3535533905932738);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_60, 0);  sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_168: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_284: "f32[38809, 12]" = torch.ops.aten.view.default(permute_168, [38809, 12]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_166], view_284, True);  view_166 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_15: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_220, permute_167, view_279]);  mul_220 = permute_167 = view_279 = None
    view_285: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_15, [3, 8, 12, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_169: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_285, [1, 3, 0, 2, 4]);  view_285 = None
    clone_103: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    view_286: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_103, [8, 197, 2304]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_287: "f32[1576, 2304]" = torch.ops.aten.view.default(view_286, [1576, 2304]);  view_286 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_287, permute_170);  permute_170 = None
    permute_171: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_171, view_163);  permute_171 = view_163 = None
    permute_172: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_61: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[2304]" = torch.ops.aten.view.default(sum_61, [2304]);  sum_61 = None
    permute_173: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_289: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_9: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 0, 768)
    slice_11: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 1536, 2304);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_222: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_93);  primals_93 = None
    mul_223: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_99);  mul_222 = None
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, sum_63);  sum_63 = None
    sub_59: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_62);  mul_223 = sum_62 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_225);  sub_59 = mul_225 = None
    mul_226: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_60);  div_19 = sub_60 = None
    mul_227: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_99);  mul_99 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_108, mul_226);  add_108 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_228: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_89);  primals_89 = None
    mul_229: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, clone_72);  clone_72 = None
    sum_66: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1], True);  mul_229 = None
    view_290: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 768]" = torch.ops.aten.view.default(mul_228, [1576, 768]);  mul_228 = None
    mm_26: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_291, permute_174);  permute_174 = None
    permute_175: "f32[768, 1576]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_175, view_161);  permute_175 = view_161 = None
    permute_176: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_177: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_293: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_26, [8, 197, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_231: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
    mul_232: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, view_160)
    mul_233: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
    exp_15: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
    mul_234: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_235: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, mul_234);  view_160 = mul_234 = None
    add_111: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
    mul_236: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_293, add_111);  view_293 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_236, [1576, 3072]);  mul_236 = None
    mm_28: "f32[1576, 768]" = torch.ops.aten.mm.default(view_294, permute_178);  permute_178 = None
    permute_179: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_179, view_159);  permute_179 = view_159 = None
    permute_180: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[3072]" = torch.ops.aten.view.default(sum_68, [3072]);  sum_68 = None
    permute_181: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_296: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_28, [8, 197, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_238: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, primals_90);  primals_90 = None
    mul_239: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_93);  mul_238 = None
    sum_70: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, sum_70);  sum_70 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_69);  mul_239 = sum_69 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_241);  sub_62 = mul_241 = None
    mul_242: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_63);  div_20 = sub_63 = None
    mul_243: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, mul_93);  mul_93 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_296, [0, 1]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_109, mul_242);  add_109 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_244: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, primals_82);  primals_82 = None
    mul_245: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, clone_70);  clone_70 = None
    sum_73: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 1], True);  mul_245 = None
    view_297: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_298: "f32[1576, 768]" = torch.ops.aten.view.default(mul_244, [1576, 768]);  mul_244 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_298, permute_182);  permute_182 = None
    permute_183: "f32[768, 1576]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_183, view_157);  permute_183 = view_157 = None
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_300: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_30, [8, 197, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_301: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_300, [8, 197, 12, 64]);  view_300 = None
    permute_186: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_104: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_302: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_104, [96, 197, 64]);  clone_104 = None
    bmm_36: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_187, view_302);  permute_187 = None
    bmm_37: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_302, permute_188);  view_302 = permute_188 = None
    view_303: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_36, [8, 12, 197, 64]);  bmm_36 = None
    view_304: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_37, [8, 12, 197, 197]);  bmm_37 = None
    mul_246: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_304, alias_15);  view_304 = None
    sum_75: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [-1], True)
    mul_247: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_15, sum_75);  alias_15 = sum_75 = None
    sub_64: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    sum_76: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_64, [0], True)
    view_305: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_64, [96, 197, 197]);  sub_64 = None
    bmm_38: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_189, view_305);  permute_189 = None
    bmm_39: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_305, permute_190);  view_305 = permute_190 = None
    view_306: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_38, [8, 12, 64, 197]);  bmm_38 = None
    view_307: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_39, [8, 12, 197, 64]);  bmm_39 = None
    mul_248: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_306, 0.3535533905932738);  view_306 = None
    permute_191: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_248, [0, 1, 3, 2]);  mul_248 = None
    mul_249: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_307, 0.3535533905932738);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_76, 0);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_192: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_308: "f32[38809, 12]" = torch.ops.aten.view.default(permute_192, [38809, 12]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_148], view_308, True);  view_148 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_16: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_249, permute_191, view_303]);  mul_249 = permute_191 = view_303 = None
    view_309: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_16, [3, 8, 12, 197, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_193: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_309, [1, 3, 0, 2, 4]);  view_309 = None
    clone_105: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_310: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_105, [8, 197, 2304]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_311: "f32[1576, 2304]" = torch.ops.aten.view.default(view_310, [1576, 2304]);  view_310 = None
    mm_32: "f32[1576, 768]" = torch.ops.aten.mm.default(view_311, permute_194);  permute_194 = None
    permute_195: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_195, view_145);  permute_195 = view_145 = None
    permute_196: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_77: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[2304]" = torch.ops.aten.view.default(sum_77, [2304]);  sum_77 = None
    permute_197: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_313: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_32, [8, 197, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_12: "f32[768]" = torch.ops.aten.slice.Tensor(view_312, 0, 0, 768)
    slice_14: "f32[768]" = torch.ops.aten.slice.Tensor(view_312, 0, 1536, 2304);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_251: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_83);  primals_83 = None
    mul_252: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
    sum_78: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_88);  mul_251 = None
    sum_79: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_88, sum_79);  sum_79 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_78);  mul_252 = sum_78 = None
    sub_67: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_254);  sub_66 = mul_254 = None
    mul_255: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_67);  div_21 = sub_67 = None
    mul_256: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_88);  mul_88 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_112, mul_255);  add_112 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_257: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_79);  primals_79 = None
    mul_258: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, clone_64);  clone_64 = None
    sum_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
    view_314: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_315: "f32[1576, 768]" = torch.ops.aten.view.default(mul_257, [1576, 768]);  mul_257 = None
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_315, permute_198);  permute_198 = None
    permute_199: "f32[768, 1576]" = torch.ops.aten.permute.default(view_315, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_199, view_143);  permute_199 = view_143 = None
    permute_200: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_317: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_260: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_261: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, view_142)
    mul_262: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
    exp_16: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
    mul_263: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_264: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, mul_263);  view_142 = mul_263 = None
    add_115: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
    mul_265: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_317, add_115);  view_317 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_318: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_265, [1576, 3072]);  mul_265 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_318, permute_202);  permute_202 = None
    permute_203: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_318, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_203, view_141);  permute_203 = view_141 = None
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[3072]" = torch.ops.aten.view.default(sum_84, [3072]);  sum_84 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    view_320: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_267: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_320, primals_80);  primals_80 = None
    mul_268: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
    sum_85: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_82);  mul_267 = None
    sum_86: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_82, sum_86);  sum_86 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_85);  mul_268 = sum_85 = None
    sub_70: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_270);  sub_69 = mul_270 = None
    mul_271: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_70);  div_22 = sub_70 = None
    mul_272: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_320, mul_82);  mul_82 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_113, mul_271);  add_113 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_273: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_72);  primals_72 = None
    mul_274: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, clone_62);  clone_62 = None
    sum_89: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1], True);  mul_274 = None
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_322: "f32[1576, 768]" = torch.ops.aten.view.default(mul_273, [1576, 768]);  mul_273 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_322, permute_206);  permute_206 = None
    permute_207: "f32[768, 1576]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_207, view_139);  permute_207 = view_139 = None
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_209: "f32[768, 768]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    view_324: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_325: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_324, [8, 197, 12, 64]);  view_324 = None
    permute_210: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_106: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_326: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_106, [96, 197, 64]);  clone_106 = None
    bmm_40: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_211, view_326);  permute_211 = None
    bmm_41: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_326, permute_212);  view_326 = permute_212 = None
    view_327: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_40, [8, 12, 197, 64]);  bmm_40 = None
    view_328: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_41, [8, 12, 197, 197]);  bmm_41 = None
    mul_275: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_328, alias_16);  view_328 = None
    sum_91: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_16, sum_91);  alias_16 = sum_91 = None
    sub_71: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    sum_92: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_71, [0], True)
    view_329: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_71, [96, 197, 197]);  sub_71 = None
    bmm_42: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_213, view_329);  permute_213 = None
    bmm_43: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_329, permute_214);  view_329 = permute_214 = None
    view_330: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_42, [8, 12, 64, 197]);  bmm_42 = None
    view_331: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_43, [8, 12, 197, 64]);  bmm_43 = None
    mul_277: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_330, 0.3535533905932738);  view_330 = None
    permute_215: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_277, [0, 1, 3, 2]);  mul_277 = None
    mul_278: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_331, 0.3535533905932738);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_92, 0);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_216: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_332: "f32[38809, 12]" = torch.ops.aten.view.default(permute_216, [38809, 12]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_130], view_332, True);  view_130 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_17: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_278, permute_215, view_327]);  mul_278 = permute_215 = view_327 = None
    view_333: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_17, [3, 8, 12, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_217: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_333, [1, 3, 0, 2, 4]);  view_333 = None
    clone_107: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    view_334: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_107, [8, 197, 2304]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_335: "f32[1576, 2304]" = torch.ops.aten.view.default(view_334, [1576, 2304]);  view_334 = None
    mm_40: "f32[1576, 768]" = torch.ops.aten.mm.default(view_335, permute_218);  permute_218 = None
    permute_219: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_219, view_127);  permute_219 = view_127 = None
    permute_220: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_93: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[2304]" = torch.ops.aten.view.default(sum_93, [2304]);  sum_93 = None
    permute_221: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_337: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_40, [8, 197, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_15: "f32[768]" = torch.ops.aten.slice.Tensor(view_336, 0, 0, 768)
    slice_17: "f32[768]" = torch.ops.aten.slice.Tensor(view_336, 0, 1536, 2304);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_280: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_337, primals_73);  primals_73 = None
    mul_281: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_280, 768)
    sum_94: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_280, mul_77);  mul_280 = None
    sum_95: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, sum_95);  sum_95 = None
    sub_73: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_281, sum_94);  mul_281 = sum_94 = None
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_283);  sub_73 = mul_283 = None
    mul_284: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_74);  div_23 = sub_74 = None
    mul_285: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_337, mul_77);  mul_77 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_337, [0, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_116, mul_284);  add_116 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_286: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_69);  primals_69 = None
    mul_287: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, clone_56);  clone_56 = None
    sum_98: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1], True);  mul_287 = None
    view_338: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_339: "f32[1576, 768]" = torch.ops.aten.view.default(mul_286, [1576, 768]);  mul_286 = None
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_339, permute_222);  permute_222 = None
    permute_223: "f32[768, 1576]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_223, view_125);  permute_223 = view_125 = None
    permute_224: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_341: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_289: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
    mul_290: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, view_124)
    mul_291: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_17: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_293: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, mul_292);  view_124 = mul_292 = None
    add_119: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_341, add_119);  view_341 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_294, [1576, 3072]);  mul_294 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_342, permute_226);  permute_226 = None
    permute_227: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_342, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_227, view_123);  permute_227 = view_123 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_100: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_342, [0], True);  view_342 = None
    view_343: "f32[3072]" = torch.ops.aten.view.default(sum_100, [3072]);  sum_100 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_344: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_296: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_344, primals_70);  primals_70 = None
    mul_297: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_296, 768)
    sum_101: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
    mul_298: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_296, mul_71);  mul_296 = None
    sum_102: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_71, sum_102);  sum_102 = None
    sub_76: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_297, sum_101);  mul_297 = sum_101 = None
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_299);  sub_76 = mul_299 = None
    mul_300: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_77);  div_24 = sub_77 = None
    mul_301: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_344, mul_71);  mul_71 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
    sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_344, [0, 1]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_117, mul_300);  add_117 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_302: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_62);  primals_62 = None
    mul_303: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, clone_54);  clone_54 = None
    sum_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1], True);  mul_303 = None
    view_345: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_346: "f32[1576, 768]" = torch.ops.aten.view.default(mul_302, [1576, 768]);  mul_302 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_346, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_346, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_121);  permute_231 = view_121 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
    view_347: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_348: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_349: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_348, [8, 197, 12, 64]);  view_348 = None
    permute_234: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_108: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    view_350: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_108, [96, 197, 64]);  clone_108 = None
    bmm_44: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_235, view_350);  permute_235 = None
    bmm_45: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_350, permute_236);  view_350 = permute_236 = None
    view_351: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_44, [8, 12, 197, 64]);  bmm_44 = None
    view_352: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_45, [8, 12, 197, 197]);  bmm_45 = None
    mul_304: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_352, alias_17);  view_352 = None
    sum_107: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [-1], True)
    mul_305: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_17, sum_107);  alias_17 = sum_107 = None
    sub_78: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    sum_108: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_78, [0], True)
    view_353: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_78, [96, 197, 197]);  sub_78 = None
    bmm_46: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_237, view_353);  permute_237 = None
    bmm_47: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_353, permute_238);  view_353 = permute_238 = None
    view_354: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_46, [8, 12, 64, 197]);  bmm_46 = None
    view_355: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_47, [8, 12, 197, 64]);  bmm_47 = None
    mul_306: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_354, 0.3535533905932738);  view_354 = None
    permute_239: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_306, [0, 1, 3, 2]);  mul_306 = None
    mul_307: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_355, 0.3535533905932738);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_108, 0);  sum_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_240: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_356: "f32[38809, 12]" = torch.ops.aten.view.default(permute_240, [38809, 12]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_112], view_356, True);  view_112 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_18: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_307, permute_239, view_351]);  mul_307 = permute_239 = view_351 = None
    view_357: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_18, [3, 8, 12, 197, 64]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_241: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_357, [1, 3, 0, 2, 4]);  view_357 = None
    clone_109: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_358: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_109, [8, 197, 2304]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_359: "f32[1576, 2304]" = torch.ops.aten.view.default(view_358, [1576, 2304]);  view_358 = None
    mm_48: "f32[1576, 768]" = torch.ops.aten.mm.default(view_359, permute_242);  permute_242 = None
    permute_243: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_243, view_109);  permute_243 = view_109 = None
    permute_244: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_109: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[2304]" = torch.ops.aten.view.default(sum_109, [2304]);  sum_109 = None
    permute_245: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_361: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_48, [8, 197, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_18: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 0, 768)
    slice_20: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 1536, 2304);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_309: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, primals_63);  primals_63 = None
    mul_310: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_309, 768)
    sum_110: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_309, mul_66);  mul_309 = None
    sum_111: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_111);  sum_111 = None
    sub_80: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_310, sum_110);  mul_310 = sum_110 = None
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_312);  sub_80 = mul_312 = None
    mul_313: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_81);  div_25 = sub_81 = None
    mul_314: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, mul_66);  mul_66 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_121: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_120, mul_313);  add_120 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_315: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_59);  primals_59 = None
    mul_316: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, clone_48);  clone_48 = None
    sum_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_363: "f32[1576, 768]" = torch.ops.aten.view.default(mul_315, [1576, 768]);  mul_315 = None
    mm_50: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_363, permute_246);  permute_246 = None
    permute_247: "f32[768, 1576]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_247, view_107);  permute_247 = view_107 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_249: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_365: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_50, [8, 197, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_318: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_319: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_320: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
    exp_18: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
    mul_321: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_322: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, mul_321);  view_106 = mul_321 = None
    add_123: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
    mul_323: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_365, add_123);  view_365 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_366: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_323, [1576, 3072]);  mul_323 = None
    mm_52: "f32[1576, 768]" = torch.ops.aten.mm.default(view_366, permute_250);  permute_250 = None
    permute_251: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_251, view_105);  permute_251 = view_105 = None
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_368: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_52, [8, 197, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_325: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_368, primals_60);  primals_60 = None
    mul_326: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
    sum_117: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_60);  mul_325 = None
    sum_118: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_118);  sum_118 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_117);  mul_326 = sum_117 = None
    sub_84: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_328);  sub_83 = mul_328 = None
    mul_329: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_84);  div_26 = sub_84 = None
    mul_330: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_368, mul_60);  mul_60 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_368, [0, 1]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_121, mul_329);  add_121 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_331: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, primals_52);  primals_52 = None
    mul_332: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, clone_46);  clone_46 = None
    sum_121: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True);  mul_332 = None
    view_369: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_370: "f32[1576, 768]" = torch.ops.aten.view.default(mul_331, [1576, 768]);  mul_331 = None
    mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_370, permute_254);  permute_254 = None
    permute_255: "f32[768, 1576]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_255, view_103);  permute_255 = view_103 = None
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_372: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_54, [8, 197, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_373: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_372, [8, 197, 12, 64]);  view_372 = None
    permute_258: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_110: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_374: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_110, [96, 197, 64]);  clone_110 = None
    bmm_48: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_259, view_374);  permute_259 = None
    bmm_49: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_374, permute_260);  view_374 = permute_260 = None
    view_375: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_48, [8, 12, 197, 64]);  bmm_48 = None
    view_376: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_49, [8, 12, 197, 197]);  bmm_49 = None
    mul_333: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_376, alias_18);  view_376 = None
    sum_123: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [-1], True)
    mul_334: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_18, sum_123);  alias_18 = sum_123 = None
    sub_85: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    sum_124: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_85, [0], True)
    view_377: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_85, [96, 197, 197]);  sub_85 = None
    bmm_50: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_261, view_377);  permute_261 = None
    bmm_51: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_377, permute_262);  view_377 = permute_262 = None
    view_378: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_50, [8, 12, 64, 197]);  bmm_50 = None
    view_379: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_51, [8, 12, 197, 64]);  bmm_51 = None
    mul_335: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_378, 0.3535533905932738);  view_378 = None
    permute_263: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_335, [0, 1, 3, 2]);  mul_335 = None
    mul_336: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_379, 0.3535533905932738);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_124, 0);  sum_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_264: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_380: "f32[38809, 12]" = torch.ops.aten.view.default(permute_264, [38809, 12]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_94], view_380, True);  view_94 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_19: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_336, permute_263, view_375]);  mul_336 = permute_263 = view_375 = None
    view_381: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_19, [3, 8, 12, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_265: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_381, [1, 3, 0, 2, 4]);  view_381 = None
    clone_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_382: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_111, [8, 197, 2304]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_383: "f32[1576, 2304]" = torch.ops.aten.view.default(view_382, [1576, 2304]);  view_382 = None
    mm_56: "f32[1576, 768]" = torch.ops.aten.mm.default(view_383, permute_266);  permute_266 = None
    permute_267: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_267, view_91);  permute_267 = view_91 = None
    permute_268: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_125: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[2304]" = torch.ops.aten.view.default(sum_125, [2304]);  sum_125 = None
    permute_269: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_385: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_56, [8, 197, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_21: "f32[768]" = torch.ops.aten.slice.Tensor(view_384, 0, 0, 768)
    slice_23: "f32[768]" = torch.ops.aten.slice.Tensor(view_384, 0, 1536, 2304);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_338: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_385, primals_53);  primals_53 = None
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_338, 768)
    sum_126: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_338, mul_55);  mul_338 = None
    sum_127: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_127);  sum_127 = None
    sub_87: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_339, sum_126);  mul_339 = sum_126 = None
    sub_88: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_341);  sub_87 = mul_341 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_88);  div_27 = sub_88 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_385, mul_55);  mul_55 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_385, [0, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_124, mul_342);  add_124 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_49);  primals_49 = None
    mul_345: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, clone_40);  clone_40 = None
    sum_130: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1], True);  mul_345 = None
    view_386: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1576, 768]" = torch.ops.aten.view.default(mul_344, [1576, 768]);  mul_344 = None
    mm_58: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_387, permute_270);  permute_270 = None
    permute_271: "f32[768, 1576]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_271, view_89);  permute_271 = view_89 = None
    permute_272: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_273: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_389: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_58, [8, 197, 3072]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_347: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
    mul_348: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_349: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
    exp_19: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_349);  mul_349 = None
    mul_350: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_351: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_350);  view_88 = mul_350 = None
    add_127: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
    mul_352: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_389, add_127);  view_389 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_352, [1576, 3072]);  mul_352 = None
    mm_60: "f32[1576, 768]" = torch.ops.aten.mm.default(view_390, permute_274);  permute_274 = None
    permute_275: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_275, view_87);  permute_275 = view_87 = None
    permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_132: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[3072]" = torch.ops.aten.view.default(sum_132, [3072]);  sum_132 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_392: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_60, [8, 197, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_354: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_392, primals_50);  primals_50 = None
    mul_355: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_354, 768)
    sum_133: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_354, mul_49);  mul_354 = None
    sum_134: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_134);  sum_134 = None
    sub_90: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_355, sum_133);  mul_355 = sum_133 = None
    sub_91: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_357);  sub_90 = mul_357 = None
    mul_358: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_91);  div_28 = sub_91 = None
    mul_359: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_392, mul_49);  mul_49 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_392, [0, 1]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_128: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, mul_358);  add_125 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_360: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_42);  primals_42 = None
    mul_361: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, clone_38);  clone_38 = None
    sum_137: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1], True);  mul_361 = None
    view_393: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_394: "f32[1576, 768]" = torch.ops.aten.view.default(mul_360, [1576, 768]);  mul_360 = None
    mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_394, permute_278);  permute_278 = None
    permute_279: "f32[768, 1576]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_279, view_85);  permute_279 = view_85 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_281: "f32[768, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_396: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_62, [8, 197, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_397: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_396, [8, 197, 12, 64]);  view_396 = None
    permute_282: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_112: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_398: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_112, [96, 197, 64]);  clone_112 = None
    bmm_52: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_283, view_398);  permute_283 = None
    bmm_53: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_398, permute_284);  view_398 = permute_284 = None
    view_399: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_52, [8, 12, 197, 64]);  bmm_52 = None
    view_400: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_53, [8, 12, 197, 197]);  bmm_53 = None
    mul_362: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_400, alias_19);  view_400 = None
    sum_139: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [-1], True)
    mul_363: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_19, sum_139);  alias_19 = sum_139 = None
    sub_92: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    sum_140: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_92, [0], True)
    view_401: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_92, [96, 197, 197]);  sub_92 = None
    bmm_54: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_285, view_401);  permute_285 = None
    bmm_55: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_401, permute_286);  view_401 = permute_286 = None
    view_402: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_54, [8, 12, 64, 197]);  bmm_54 = None
    view_403: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_55, [8, 12, 197, 64]);  bmm_55 = None
    mul_364: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_402, 0.3535533905932738);  view_402 = None
    permute_287: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_364, [0, 1, 3, 2]);  mul_364 = None
    mul_365: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_403, 0.3535533905932738);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_140, 0);  sum_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_288: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_404: "f32[38809, 12]" = torch.ops.aten.view.default(permute_288, [38809, 12]);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_76], view_404, True);  view_76 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_20: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_365, permute_287, view_399]);  mul_365 = permute_287 = view_399 = None
    view_405: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_20, [3, 8, 12, 197, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_405, [1, 3, 0, 2, 4]);  view_405 = None
    clone_113: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_406: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_113, [8, 197, 2304]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_407: "f32[1576, 2304]" = torch.ops.aten.view.default(view_406, [1576, 2304]);  view_406 = None
    mm_64: "f32[1576, 768]" = torch.ops.aten.mm.default(view_407, permute_290);  permute_290 = None
    permute_291: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_291, view_73);  permute_291 = view_73 = None
    permute_292: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_141: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[2304]" = torch.ops.aten.view.default(sum_141, [2304]);  sum_141 = None
    permute_293: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_409: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_64, [8, 197, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_24: "f32[768]" = torch.ops.aten.slice.Tensor(view_408, 0, 0, 768)
    slice_26: "f32[768]" = torch.ops.aten.slice.Tensor(view_408, 0, 1536, 2304);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_367: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_409, primals_43);  primals_43 = None
    mul_368: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
    sum_142: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_44);  mul_367 = None
    sum_143: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_143);  sum_143 = None
    sub_94: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_142);  mul_368 = sum_142 = None
    sub_95: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_370);  sub_94 = mul_370 = None
    mul_371: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_95);  div_29 = sub_95 = None
    mul_372: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_409, mul_44);  mul_44 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_409, [0, 1]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_128, mul_371);  add_128 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_373: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_39);  primals_39 = None
    mul_374: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, clone_32);  clone_32 = None
    sum_146: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1], True);  mul_374 = None
    view_410: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_411: "f32[1576, 768]" = torch.ops.aten.view.default(mul_373, [1576, 768]);  mul_373 = None
    mm_66: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_411, permute_294);  permute_294 = None
    permute_295: "f32[768, 1576]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_295, view_71);  permute_295 = view_71 = None
    permute_296: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_297: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_413: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_66, [8, 197, 3072]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_376: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_377: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, view_70)
    mul_378: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
    exp_20: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
    mul_379: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_380: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, mul_379);  view_70 = mul_379 = None
    add_131: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
    mul_381: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_413, add_131);  view_413 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_414: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_381, [1576, 3072]);  mul_381 = None
    mm_68: "f32[1576, 768]" = torch.ops.aten.mm.default(view_414, permute_298);  permute_298 = None
    permute_299: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_299, view_69);  permute_299 = view_69 = None
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    permute_301: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_416: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_68, [8, 197, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_383: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_416, primals_40);  primals_40 = None
    mul_384: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_383, 768)
    sum_149: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_383, mul_38);  mul_383 = None
    sum_150: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_150);  sum_150 = None
    sub_97: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_384, sum_149);  mul_384 = sum_149 = None
    sub_98: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_386);  sub_97 = mul_386 = None
    mul_387: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_98);  div_30 = sub_98 = None
    mul_388: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_416, mul_38);  mul_38 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_416, [0, 1]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_387);  add_129 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_389: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_32);  primals_32 = None
    mul_390: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, clone_30);  clone_30 = None
    sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1], True);  mul_390 = None
    view_417: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_418: "f32[1576, 768]" = torch.ops.aten.view.default(mul_389, [1576, 768]);  mul_389 = None
    mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_418, permute_302);  permute_302 = None
    permute_303: "f32[768, 1576]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_303, view_67);  permute_303 = view_67 = None
    permute_304: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
    view_419: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_420: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_70, [8, 197, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_421: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_420, [8, 197, 12, 64]);  view_420 = None
    permute_306: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_114: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    view_422: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_114, [96, 197, 64]);  clone_114 = None
    bmm_56: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_307, view_422);  permute_307 = None
    bmm_57: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_422, permute_308);  view_422 = permute_308 = None
    view_423: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_56, [8, 12, 197, 64]);  bmm_56 = None
    view_424: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_57, [8, 12, 197, 197]);  bmm_57 = None
    mul_391: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_424, alias_20);  view_424 = None
    sum_155: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [-1], True)
    mul_392: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_20, sum_155);  alias_20 = sum_155 = None
    sub_99: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    sum_156: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_99, [0], True)
    view_425: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_99, [96, 197, 197]);  sub_99 = None
    bmm_58: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_309, view_425);  permute_309 = None
    bmm_59: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_425, permute_310);  view_425 = permute_310 = None
    view_426: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_58, [8, 12, 64, 197]);  bmm_58 = None
    view_427: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_59, [8, 12, 197, 64]);  bmm_59 = None
    mul_393: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_426, 0.3535533905932738);  view_426 = None
    permute_311: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_393, [0, 1, 3, 2]);  mul_393 = None
    mul_394: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_427, 0.3535533905932738);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_156, 0);  sum_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_312: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_428: "f32[38809, 12]" = torch.ops.aten.view.default(permute_312, [38809, 12]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_58], view_428, True);  view_58 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_21: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_394, permute_311, view_423]);  mul_394 = permute_311 = view_423 = None
    view_429: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 12, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_313: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_429, [1, 3, 0, 2, 4]);  view_429 = None
    clone_115: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
    view_430: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_115, [8, 197, 2304]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_431: "f32[1576, 2304]" = torch.ops.aten.view.default(view_430, [1576, 2304]);  view_430 = None
    mm_72: "f32[1576, 768]" = torch.ops.aten.mm.default(view_431, permute_314);  permute_314 = None
    permute_315: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_315, view_55);  permute_315 = view_55 = None
    permute_316: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_157: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[2304]" = torch.ops.aten.view.default(sum_157, [2304]);  sum_157 = None
    permute_317: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_433: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_72, [8, 197, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_27: "f32[768]" = torch.ops.aten.slice.Tensor(view_432, 0, 0, 768)
    slice_29: "f32[768]" = torch.ops.aten.slice.Tensor(view_432, 0, 1536, 2304);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_396: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_433, primals_33);  primals_33 = None
    mul_397: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_396, 768)
    sum_158: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_396, mul_33);  mul_396 = None
    sum_159: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_159);  sum_159 = None
    sub_101: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_397, sum_158);  mul_397 = sum_158 = None
    sub_102: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_399);  sub_101 = mul_399 = None
    mul_400: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_102);  div_31 = sub_102 = None
    mul_401: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_433, mul_33);  mul_33 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_133: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_132, mul_400);  add_132 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_402: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_29);  primals_29 = None
    mul_403: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_133, clone_24);  clone_24 = None
    sum_162: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1], True);  mul_403 = None
    view_434: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_435: "f32[1576, 768]" = torch.ops.aten.view.default(mul_402, [1576, 768]);  mul_402 = None
    mm_74: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_435, permute_318);  permute_318 = None
    permute_319: "f32[768, 1576]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_319, view_53);  permute_319 = view_53 = None
    permute_320: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_321: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_437: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_74, [8, 197, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_405: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
    mul_406: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, view_52)
    mul_407: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_21: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_409: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, mul_408);  view_52 = mul_408 = None
    add_135: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_437, add_135);  view_437 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_438: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_410, [1576, 3072]);  mul_410 = None
    mm_76: "f32[1576, 768]" = torch.ops.aten.mm.default(view_438, permute_322);  permute_322 = None
    permute_323: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_323, view_51);  permute_323 = view_51 = None
    permute_324: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_164: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
    view_439: "f32[3072]" = torch.ops.aten.view.default(sum_164, [3072]);  sum_164 = None
    permute_325: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    view_440: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_76, [8, 197, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_412: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_440, primals_30);  primals_30 = None
    mul_413: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
    sum_165: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_27);  mul_412 = None
    sum_166: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, sum_166);  sum_166 = None
    sub_104: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_165);  mul_413 = sum_165 = None
    sub_105: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_415);  sub_104 = mul_415 = None
    mul_416: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_105);  div_32 = sub_105 = None
    mul_417: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_440, mul_27);  mul_27 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_440, [0, 1]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_136: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_133, mul_416);  add_133 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_418: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_136, primals_22);  primals_22 = None
    mul_419: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_136, clone_22);  clone_22 = None
    sum_169: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 1], True);  mul_419 = None
    view_441: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_442: "f32[1576, 768]" = torch.ops.aten.view.default(mul_418, [1576, 768]);  mul_418 = None
    mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_442, permute_326);  permute_326 = None
    permute_327: "f32[768, 1576]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_327, view_49);  permute_327 = view_49 = None
    permute_328: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
    view_444: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_78, [8, 197, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_445: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_444, [8, 197, 12, 64]);  view_444 = None
    permute_330: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_116: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
    view_446: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_116, [96, 197, 64]);  clone_116 = None
    bmm_60: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_331, view_446);  permute_331 = None
    bmm_61: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_446, permute_332);  view_446 = permute_332 = None
    view_447: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_60, [8, 12, 197, 64]);  bmm_60 = None
    view_448: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_61, [8, 12, 197, 197]);  bmm_61 = None
    mul_420: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_448, alias_21);  view_448 = None
    sum_171: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [-1], True)
    mul_421: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_21, sum_171);  alias_21 = sum_171 = None
    sub_106: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    sum_172: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_106, [0], True)
    view_449: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_106, [96, 197, 197]);  sub_106 = None
    bmm_62: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_333, view_449);  permute_333 = None
    bmm_63: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_449, permute_334);  view_449 = permute_334 = None
    view_450: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_62, [8, 12, 64, 197]);  bmm_62 = None
    view_451: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_63, [8, 12, 197, 64]);  bmm_63 = None
    mul_422: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_450, 0.3535533905932738);  view_450 = None
    permute_335: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_422, [0, 1, 3, 2]);  mul_422 = None
    mul_423: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_451, 0.3535533905932738);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_172, 0);  sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_336: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_452: "f32[38809, 12]" = torch.ops.aten.view.default(permute_336, [38809, 12]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_40], view_452, True);  view_40 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_22: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_423, permute_335, view_447]);  mul_423 = permute_335 = view_447 = None
    view_453: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_22, [3, 8, 12, 197, 64]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_337: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_453, [1, 3, 0, 2, 4]);  view_453 = None
    clone_117: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    view_454: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_117, [8, 197, 2304]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_455: "f32[1576, 2304]" = torch.ops.aten.view.default(view_454, [1576, 2304]);  view_454 = None
    mm_80: "f32[1576, 768]" = torch.ops.aten.mm.default(view_455, permute_338);  permute_338 = None
    permute_339: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_339, view_37);  permute_339 = view_37 = None
    permute_340: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_173: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
    view_456: "f32[2304]" = torch.ops.aten.view.default(sum_173, [2304]);  sum_173 = None
    permute_341: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_457: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_80, [8, 197, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_30: "f32[768]" = torch.ops.aten.slice.Tensor(view_456, 0, 0, 768)
    slice_32: "f32[768]" = torch.ops.aten.slice.Tensor(view_456, 0, 1536, 2304);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_425: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_457, primals_23);  primals_23 = None
    mul_426: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_425, 768)
    sum_174: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_22);  mul_425 = None
    sum_175: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_175);  sum_175 = None
    sub_108: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_426, sum_174);  mul_426 = sum_174 = None
    sub_109: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_428);  sub_108 = mul_428 = None
    mul_429: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_109);  div_33 = sub_109 = None
    mul_430: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_457, mul_22);  mul_22 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_457, [0, 1]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_137: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_136, mul_429);  add_136 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_431: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_19);  primals_19 = None
    mul_432: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_137, clone_16);  clone_16 = None
    sum_178: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1], True);  mul_432 = None
    view_458: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[1576, 768]" = torch.ops.aten.view.default(mul_431, [1576, 768]);  mul_431 = None
    mm_82: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_459, permute_342);  permute_342 = None
    permute_343: "f32[768, 1576]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_343, view_35);  permute_343 = view_35 = None
    permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_345: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    view_461: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_82, [8, 197, 3072]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_434: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
    mul_435: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, view_34)
    mul_436: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
    exp_22: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_436);  mul_436 = None
    mul_437: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_438: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, mul_437);  view_34 = mul_437 = None
    add_139: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
    mul_439: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_461, add_139);  view_461 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_439, [1576, 3072]);  mul_439 = None
    mm_84: "f32[1576, 768]" = torch.ops.aten.mm.default(view_462, permute_346);  permute_346 = None
    permute_347: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_347, view_33);  permute_347 = view_33 = None
    permute_348: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_180: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[3072]" = torch.ops.aten.view.default(sum_180, [3072]);  sum_180 = None
    permute_349: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_464: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_84, [8, 197, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_441: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_464, primals_20);  primals_20 = None
    mul_442: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_441, 768)
    sum_181: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
    mul_443: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_441, mul_16);  mul_441 = None
    sum_182: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
    mul_444: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_182);  sum_182 = None
    sub_111: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_442, sum_181);  mul_442 = sum_181 = None
    sub_112: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_444);  sub_111 = mul_444 = None
    mul_445: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_112);  div_34 = sub_112 = None
    mul_446: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_464, mul_16);  mul_16 = None
    sum_183: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_464, [0, 1]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_137, mul_445);  add_137 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_447: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_12);  primals_12 = None
    mul_448: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_140, clone_14);  clone_14 = None
    sum_185: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1], True);  mul_448 = None
    view_465: "f32[768]" = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_466: "f32[1576, 768]" = torch.ops.aten.view.default(mul_447, [1576, 768]);  mul_447 = None
    mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_466, permute_350);  permute_350 = None
    permute_351: "f32[768, 1576]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_351, view_31);  permute_351 = view_31 = None
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_353: "f32[768, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_468: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_86, [8, 197, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_469: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_468, [8, 197, 12, 64]);  view_468 = None
    permute_354: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_118: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_470: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_118, [96, 197, 64]);  clone_118 = None
    bmm_64: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_355, view_470);  permute_355 = None
    bmm_65: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_470, permute_356);  view_470 = permute_356 = None
    view_471: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_64, [8, 12, 197, 64]);  bmm_64 = None
    view_472: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_65, [8, 12, 197, 197]);  bmm_65 = None
    mul_449: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_472, alias_22);  view_472 = None
    sum_187: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [-1], True)
    mul_450: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_22, sum_187);  alias_22 = sum_187 = None
    sub_113: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    sum_188: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_113, [0], True)
    view_473: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_113, [96, 197, 197]);  sub_113 = None
    bmm_66: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_357, view_473);  permute_357 = None
    bmm_67: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_473, permute_358);  view_473 = permute_358 = None
    view_474: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_66, [8, 12, 64, 197]);  bmm_66 = None
    view_475: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_67, [8, 12, 197, 64]);  bmm_67 = None
    mul_451: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_474, 0.3535533905932738);  view_474 = None
    permute_359: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_451, [0, 1, 3, 2]);  mul_451 = None
    mul_452: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_475, 0.3535533905932738);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_188, 0);  sum_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_360: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_476: "f32[38809, 12]" = torch.ops.aten.view.default(permute_360, [38809, 12]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_22], view_476, True);  view_22 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_23: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_452, permute_359, view_471]);  mul_452 = permute_359 = view_471 = None
    view_477: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 12, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_361: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_477, [1, 3, 0, 2, 4]);  view_477 = None
    clone_119: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_478: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_119, [8, 197, 2304]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_479: "f32[1576, 2304]" = torch.ops.aten.view.default(view_478, [1576, 2304]);  view_478 = None
    mm_88: "f32[1576, 768]" = torch.ops.aten.mm.default(view_479, permute_362);  permute_362 = None
    permute_363: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_363, view_19);  permute_363 = view_19 = None
    permute_364: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_189: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[2304]" = torch.ops.aten.view.default(sum_189, [2304]);  sum_189 = None
    permute_365: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_481: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_88, [8, 197, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_33: "f32[768]" = torch.ops.aten.slice.Tensor(view_480, 0, 0, 768)
    slice_35: "f32[768]" = torch.ops.aten.slice.Tensor(view_480, 0, 1536, 2304);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_454: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_481, primals_13);  primals_13 = None
    mul_455: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_454, 768)
    sum_190: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_454, mul_11);  mul_454 = None
    sum_191: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_11, sum_191);  sum_191 = None
    sub_115: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_455, sum_190);  mul_455 = sum_190 = None
    sub_116: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_457);  sub_115 = mul_457 = None
    mul_458: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_116);  div_35 = sub_116 = None
    mul_459: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_481, mul_11);  mul_11 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_481, [0, 1]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_141: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_140, mul_458);  add_140 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_460: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_9);  primals_9 = None
    mul_461: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_141, clone_8);  clone_8 = None
    sum_194: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1], True);  mul_461 = None
    view_482: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_483: "f32[1576, 768]" = torch.ops.aten.view.default(mul_460, [1576, 768]);  mul_460 = None
    mm_90: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_483, permute_366);  permute_366 = None
    permute_367: "f32[768, 1576]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_367, view_17);  permute_367 = view_17 = None
    permute_368: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
    view_484: "f32[768]" = torch.ops.aten.view.default(sum_195, [768]);  sum_195 = None
    permute_369: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_485: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_90, [8, 197, 3072]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_463: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_464: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, view_16)
    mul_465: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
    exp_23: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_465);  mul_465 = None
    mul_466: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_467: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, mul_466);  view_16 = mul_466 = None
    add_143: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
    mul_468: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_485, add_143);  view_485 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_486: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_468, [1576, 3072]);  mul_468 = None
    mm_92: "f32[1576, 768]" = torch.ops.aten.mm.default(view_486, permute_370);  permute_370 = None
    permute_371: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_371, view_15);  permute_371 = view_15 = None
    permute_372: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_196: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[3072]" = torch.ops.aten.view.default(sum_196, [3072]);  sum_196 = None
    permute_373: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_488: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_92, [8, 197, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_470: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_488, primals_10);  primals_10 = None
    mul_471: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_470, 768)
    sum_197: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
    mul_472: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_470, mul_5);  mul_470 = None
    sum_198: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
    mul_473: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_5, sum_198);  sum_198 = None
    sub_118: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_471, sum_197);  mul_471 = sum_197 = None
    sub_119: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_473);  sub_118 = mul_473 = None
    mul_474: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_119);  div_36 = sub_119 = None
    mul_475: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_488, mul_5);  mul_5 = None
    sum_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_488, [0, 1]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_144: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_141, mul_474);  add_141 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_476: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_2);  primals_2 = None
    mul_477: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_144, clone_6);  clone_6 = None
    sum_201: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 1], True);  mul_477 = None
    view_489: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_490: "f32[1576, 768]" = torch.ops.aten.view.default(mul_476, [1576, 768]);  mul_476 = None
    mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_490, permute_374);  permute_374 = None
    permute_375: "f32[768, 1576]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_375, view_13);  permute_375 = view_13 = None
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_202, [768]);  sum_202 = None
    permute_377: "f32[768, 768]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_492: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_94, [8, 197, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_493: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_492, [8, 197, 12, 64]);  view_492 = None
    permute_378: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_120: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
    view_494: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_120, [96, 197, 64]);  clone_120 = None
    bmm_68: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_379, view_494);  permute_379 = None
    bmm_69: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_494, permute_380);  view_494 = permute_380 = None
    view_495: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_68, [8, 12, 197, 64]);  bmm_68 = None
    view_496: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_69, [8, 12, 197, 197]);  bmm_69 = None
    mul_478: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_496, alias_23);  view_496 = None
    sum_203: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
    mul_479: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_23, sum_203);  alias_23 = sum_203 = None
    sub_120: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    sum_204: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_120, [0], True)
    view_497: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_120, [96, 197, 197]);  sub_120 = None
    bmm_70: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_381, view_497);  permute_381 = None
    bmm_71: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_497, permute_382);  view_497 = permute_382 = None
    view_498: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_70, [8, 12, 64, 197]);  bmm_70 = None
    view_499: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_71, [8, 12, 197, 64]);  bmm_71 = None
    mul_480: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_498, 0.3535533905932738);  view_498 = None
    permute_383: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_480, [0, 1, 3, 2]);  mul_480 = None
    mul_481: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_499, 0.3535533905932738);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_204, 0);  sum_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_384: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_500: "f32[38809, 12]" = torch.ops.aten.view.default(permute_384, [38809, 12]);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_put_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_default_2, [view_4], view_500, True);  full_default_2 = view_4 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_24: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_481, permute_383, view_495]);  mul_481 = permute_383 = view_495 = None
    view_501: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_24, [3, 8, 12, 197, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_385: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_501, [1, 3, 0, 2, 4]);  view_501 = None
    clone_121: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_502: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_121, [8, 197, 2304]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_503: "f32[1576, 2304]" = torch.ops.aten.view.default(view_502, [1576, 2304]);  view_502 = None
    mm_96: "f32[1576, 768]" = torch.ops.aten.mm.default(view_503, permute_386);  permute_386 = None
    permute_387: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_387, view_1);  permute_387 = view_1 = None
    permute_388: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_205: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[2304]" = torch.ops.aten.view.default(sum_205, [2304]);  sum_205 = None
    permute_389: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    view_505: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_96, [8, 197, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_36: "f32[768]" = torch.ops.aten.slice.Tensor(view_504, 0, 0, 768)
    slice_38: "f32[768]" = torch.ops.aten.slice.Tensor(view_504, 0, 1536, 2304);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_483: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_505, primals_3);  primals_3 = None
    mul_484: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_483, 768)
    sum_206: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_483, mul);  mul_483 = None
    sum_207: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, sum_207);  sum_207 = None
    sub_122: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_484, sum_206);  mul_484 = sum_206 = None
    sub_123: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_486);  sub_122 = mul_486 = None
    div_37: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_487: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_123);  div_37 = sub_123 = None
    mul_488: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_505, mul);  mul = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_505, [0, 1]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_145: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_144, mul_487);  add_144 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    slice_39: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_145, 1, 0, 1)
    slice_40: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_145, 1, 1, 197);  add_145 = None
    sum_210: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_39, [0], True);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_390: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_40, [0, 2, 1]);  slice_40 = None
    view_506: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_390, [8, 768, 14, 14]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_506, primals_224, primals_124, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_506 = primals_224 = primals_124 = None
    getitem_87: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_88: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return [sum_210, view_489, sum_208, sum_209, slice_36, slice_38, permute_389, index_put_11, view_482, sum_199, sum_200, view_465, sum_192, sum_193, slice_33, slice_35, permute_365, index_put_10, view_458, sum_183, sum_184, view_441, sum_176, sum_177, slice_30, slice_32, permute_341, index_put_9, view_434, sum_167, sum_168, view_417, sum_160, sum_161, slice_27, slice_29, permute_317, index_put_8, view_410, sum_151, sum_152, view_393, sum_144, sum_145, slice_24, slice_26, permute_293, index_put_7, view_386, sum_135, sum_136, view_369, sum_128, sum_129, slice_21, slice_23, permute_269, index_put_6, view_362, sum_119, sum_120, view_345, sum_112, sum_113, slice_18, slice_20, permute_245, index_put_5, view_338, sum_103, sum_104, view_321, sum_96, sum_97, slice_15, slice_17, permute_221, index_put_4, view_314, sum_87, sum_88, view_297, sum_80, sum_81, slice_12, slice_14, permute_197, index_put_3, view_290, sum_71, sum_72, view_273, sum_64, sum_65, slice_9, slice_11, permute_173, index_put_2, view_266, sum_55, sum_56, view_249, sum_48, sum_49, slice_6, slice_8, permute_149, index_put_1, view_242, sum_39, sum_40, view_225, sum_32, sum_33, slice_3, slice_5, permute_125, index_put, view_218, sum_23, sum_24, sum_16, sum_17, getitem_87, getitem_88, permute_377, view_491, permute_373, view_487, permute_369, view_484, permute_353, view_467, permute_349, view_463, permute_345, view_460, permute_329, view_443, permute_325, view_439, permute_321, view_436, permute_305, view_419, permute_301, view_415, permute_297, view_412, permute_281, view_395, permute_277, view_391, permute_273, view_388, permute_257, view_371, permute_253, view_367, permute_249, view_364, permute_233, view_347, permute_229, view_343, permute_225, view_340, permute_209, view_323, permute_205, view_319, permute_201, view_316, permute_185, view_299, permute_181, view_295, permute_177, view_292, permute_161, view_275, permute_157, view_271, permute_153, view_268, permute_137, view_251, permute_133, view_247, permute_129, view_244, permute_113, view_227, permute_109, view_223, permute_105, view_220, permute_101, view_217, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    