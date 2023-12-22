from __future__ import annotations



def forward(self, primals_3: "f32[256, 3, 14, 14]", primals_5: "f32[256]", primals_11: "f32[256]", primals_17: "f32[256]", primals_23: "f32[256]", primals_29: "f32[256]", primals_35: "f32[256]", primals_41: "f32[512, 1, 3, 3]", primals_45: "f32[512]", primals_51: "f32[512]", primals_57: "f32[512]", primals_63: "f32[512]", primals_69: "f32[512]", primals_75: "f32[512]", primals_81: "f32[512]", primals_87: "f32[512]", primals_93: "f32[512]", primals_99: "f32[512]", primals_105: "f32[512]", primals_111: "f32[512]", primals_117: "f32[1024, 1, 3, 3]", primals_121: "f32[1024]", primals_127: "f32[1024]", primals_133: "f32[1024]", primals_139: "f32[1024]", primals_145: "f32[1024]", primals_151: "f32[1024]", primals_157: "f32[1024]", primals_163: "f32[1024]", primals_169: "f32[1024]", primals_173: "f32[8, 3, 224, 224]", cat: "f32[8, 962, 256]", getitem_1: "f32[8, 962, 1]", rsqrt: "f32[8, 962, 1]", view_1: "f32[7696, 256]", getitem_2: "f32[8, 4, 962, 64]", getitem_3: "f32[8, 4, 962, 64]", getitem_4: "f32[8, 4, 962, 64]", getitem_6: "f32[8, 4, 962]", getitem_7: "i32[]", getitem_8: "i32[]", getitem_11: "i64[]", getitem_12: "i64[]", view_5: "f32[7696, 256]", mul_2: "f32[8, 962, 256]", view_7: "f32[7696, 256]", addmm_2: "f32[7696, 1024]", view_9: "f32[7696, 1024]", mul_7: "f32[8, 962, 256]", view_11: "f32[7696, 256]", getitem_18: "f32[8, 4, 962, 64]", getitem_19: "f32[8, 4, 962, 64]", getitem_20: "f32[8, 4, 962, 64]", getitem_22: "f32[8, 4, 962]", getitem_23: "i32[]", getitem_24: "i32[]", getitem_27: "i64[]", getitem_28: "i64[]", view_15: "f32[7696, 256]", mul_9: "f32[8, 962, 256]", view_17: "f32[7696, 256]", addmm_6: "f32[7696, 1024]", view_19: "f32[7696, 1024]", mul_14: "f32[8, 962, 256]", view_21: "f32[7696, 256]", getitem_34: "f32[8, 4, 962, 64]", getitem_35: "f32[8, 4, 962, 64]", getitem_36: "f32[8, 4, 962, 64]", getitem_38: "f32[8, 4, 962]", getitem_39: "i32[]", getitem_40: "i32[]", getitem_43: "i64[]", getitem_44: "i64[]", view_25: "f32[7696, 256]", mul_16: "f32[8, 962, 256]", view_27: "f32[7696, 256]", addmm_10: "f32[7696, 1024]", view_29: "f32[7696, 1024]", view_31: "f32[8, 256, 31, 31]", view_32: "f32[8, 256]", cat_1: "f32[8, 257, 512]", getitem_49: "f32[8, 257, 1]", rsqrt_6: "f32[8, 257, 1]", view_35: "f32[2056, 512]", getitem_50: "f32[8, 8, 257, 64]", getitem_51: "f32[8, 8, 257, 64]", getitem_52: "f32[8, 8, 257, 64]", getitem_54: "f32[8, 8, 257]", getitem_55: "i32[]", getitem_56: "i32[]", getitem_59: "i64[]", getitem_60: "i64[]", view_39: "f32[2056, 512]", mul_23: "f32[8, 257, 512]", view_41: "f32[2056, 512]", addmm_14: "f32[2056, 2048]", view_43: "f32[2056, 2048]", mul_28: "f32[8, 257, 512]", view_45: "f32[2056, 512]", getitem_66: "f32[8, 8, 257, 64]", getitem_67: "f32[8, 8, 257, 64]", getitem_68: "f32[8, 8, 257, 64]", getitem_70: "f32[8, 8, 257]", getitem_71: "i32[]", getitem_72: "i32[]", getitem_75: "i64[]", getitem_76: "i64[]", view_49: "f32[2056, 512]", mul_30: "f32[8, 257, 512]", view_51: "f32[2056, 512]", addmm_18: "f32[2056, 2048]", view_53: "f32[2056, 2048]", mul_35: "f32[8, 257, 512]", view_55: "f32[2056, 512]", getitem_82: "f32[8, 8, 257, 64]", getitem_83: "f32[8, 8, 257, 64]", getitem_84: "f32[8, 8, 257, 64]", getitem_86: "f32[8, 8, 257]", getitem_87: "i32[]", getitem_88: "i32[]", getitem_91: "i64[]", getitem_92: "i64[]", view_59: "f32[2056, 512]", mul_37: "f32[8, 257, 512]", view_61: "f32[2056, 512]", addmm_22: "f32[2056, 2048]", view_63: "f32[2056, 2048]", mul_42: "f32[8, 257, 512]", view_65: "f32[2056, 512]", getitem_98: "f32[8, 8, 257, 64]", getitem_99: "f32[8, 8, 257, 64]", getitem_100: "f32[8, 8, 257, 64]", getitem_102: "f32[8, 8, 257]", getitem_103: "i32[]", getitem_104: "i32[]", getitem_107: "i64[]", getitem_108: "i64[]", view_69: "f32[2056, 512]", mul_44: "f32[8, 257, 512]", view_71: "f32[2056, 512]", addmm_26: "f32[2056, 2048]", view_73: "f32[2056, 2048]", mul_49: "f32[8, 257, 512]", view_75: "f32[2056, 512]", getitem_114: "f32[8, 8, 257, 64]", getitem_115: "f32[8, 8, 257, 64]", getitem_116: "f32[8, 8, 257, 64]", getitem_118: "f32[8, 8, 257]", getitem_119: "i32[]", getitem_120: "i32[]", getitem_123: "i64[]", getitem_124: "i64[]", view_79: "f32[2056, 512]", mul_51: "f32[8, 257, 512]", view_81: "f32[2056, 512]", addmm_30: "f32[2056, 2048]", view_83: "f32[2056, 2048]", mul_56: "f32[8, 257, 512]", view_85: "f32[2056, 512]", getitem_130: "f32[8, 8, 257, 64]", getitem_131: "f32[8, 8, 257, 64]", getitem_132: "f32[8, 8, 257, 64]", getitem_134: "f32[8, 8, 257]", getitem_135: "i32[]", getitem_136: "i32[]", getitem_139: "i64[]", getitem_140: "i64[]", view_89: "f32[2056, 512]", mul_58: "f32[8, 257, 512]", view_91: "f32[2056, 512]", addmm_34: "f32[2056, 2048]", view_93: "f32[2056, 2048]", view_95: "f32[8, 512, 16, 16]", view_96: "f32[8, 512]", cat_2: "f32[8, 65, 1024]", getitem_145: "f32[8, 65, 1]", rsqrt_18: "f32[8, 65, 1]", view_99: "f32[520, 1024]", getitem_146: "f32[8, 16, 65, 64]", getitem_147: "f32[8, 16, 65, 64]", getitem_148: "f32[8, 16, 65, 64]", getitem_150: "f32[8, 16, 65]", getitem_151: "i32[]", getitem_152: "i32[]", getitem_155: "i64[]", getitem_156: "i64[]", view_103: "f32[520, 1024]", mul_65: "f32[8, 65, 1024]", view_105: "f32[520, 1024]", addmm_38: "f32[520, 4096]", view_107: "f32[520, 4096]", mul_70: "f32[8, 65, 1024]", view_109: "f32[520, 1024]", getitem_162: "f32[8, 16, 65, 64]", getitem_163: "f32[8, 16, 65, 64]", getitem_164: "f32[8, 16, 65, 64]", getitem_166: "f32[8, 16, 65]", getitem_167: "i32[]", getitem_168: "i32[]", getitem_171: "i64[]", getitem_172: "i64[]", view_113: "f32[520, 1024]", mul_72: "f32[8, 65, 1024]", view_115: "f32[520, 1024]", addmm_42: "f32[520, 4096]", view_117: "f32[520, 4096]", mul_77: "f32[8, 65, 1024]", view_119: "f32[520, 1024]", getitem_178: "f32[8, 16, 65, 64]", getitem_179: "f32[8, 16, 65, 64]", getitem_180: "f32[8, 16, 65, 64]", getitem_182: "f32[8, 16, 65]", getitem_183: "i32[]", getitem_184: "i32[]", getitem_187: "i64[]", getitem_188: "i64[]", view_123: "f32[520, 1024]", mul_79: "f32[8, 65, 1024]", view_125: "f32[520, 1024]", addmm_46: "f32[520, 4096]", view_127: "f32[520, 4096]", mul_84: "f32[8, 65, 1024]", view_129: "f32[520, 1024]", getitem_194: "f32[8, 16, 65, 64]", getitem_195: "f32[8, 16, 65, 64]", getitem_196: "f32[8, 16, 65, 64]", getitem_198: "f32[8, 16, 65]", getitem_199: "i32[]", getitem_200: "i32[]", getitem_203: "i64[]", getitem_204: "i64[]", view_133: "f32[520, 1024]", mul_86: "f32[8, 65, 1024]", view_135: "f32[520, 1024]", addmm_50: "f32[520, 4096]", view_137: "f32[520, 4096]", mul_91: "f32[8, 1, 1024]", clone_41: "f32[8, 1024]", permute_87: "f32[1000, 1024]", div: "f32[8, 1, 1]", permute_91: "f32[1024, 4096]", permute_95: "f32[4096, 1024]", div_1: "f32[8, 65, 1]", permute_99: "f32[1024, 1024]", alias_13: "f32[8, 16, 65, 64]", permute_105: "f32[3072, 1024]", div_2: "f32[8, 65, 1]", permute_109: "f32[1024, 4096]", permute_113: "f32[4096, 1024]", div_3: "f32[8, 65, 1]", permute_117: "f32[1024, 1024]", alias_14: "f32[8, 16, 65, 64]", permute_123: "f32[3072, 1024]", div_4: "f32[8, 65, 1]", permute_127: "f32[1024, 4096]", permute_131: "f32[4096, 1024]", div_5: "f32[8, 65, 1]", permute_135: "f32[1024, 1024]", alias_15: "f32[8, 16, 65, 64]", permute_141: "f32[3072, 1024]", div_6: "f32[8, 65, 1]", permute_145: "f32[1024, 4096]", permute_149: "f32[4096, 1024]", div_7: "f32[8, 65, 1]", permute_153: "f32[1024, 1024]", alias_16: "f32[8, 16, 65, 64]", permute_159: "f32[3072, 1024]", permute_166: "f32[1024, 512]", permute_169: "f32[512, 2048]", permute_173: "f32[2048, 512]", div_9: "f32[8, 257, 1]", permute_177: "f32[512, 512]", alias_17: "f32[8, 8, 257, 64]", permute_183: "f32[1536, 512]", div_10: "f32[8, 257, 1]", permute_187: "f32[512, 2048]", permute_191: "f32[2048, 512]", div_11: "f32[8, 257, 1]", permute_195: "f32[512, 512]", alias_18: "f32[8, 8, 257, 64]", permute_201: "f32[1536, 512]", div_12: "f32[8, 257, 1]", permute_205: "f32[512, 2048]", permute_209: "f32[2048, 512]", div_13: "f32[8, 257, 1]", permute_213: "f32[512, 512]", alias_19: "f32[8, 8, 257, 64]", permute_219: "f32[1536, 512]", div_14: "f32[8, 257, 1]", permute_223: "f32[512, 2048]", permute_227: "f32[2048, 512]", div_15: "f32[8, 257, 1]", permute_231: "f32[512, 512]", alias_20: "f32[8, 8, 257, 64]", permute_237: "f32[1536, 512]", div_16: "f32[8, 257, 1]", permute_241: "f32[512, 2048]", permute_245: "f32[2048, 512]", div_17: "f32[8, 257, 1]", permute_249: "f32[512, 512]", alias_21: "f32[8, 8, 257, 64]", permute_255: "f32[1536, 512]", div_18: "f32[8, 257, 1]", permute_259: "f32[512, 2048]", permute_263: "f32[2048, 512]", div_19: "f32[8, 257, 1]", permute_267: "f32[512, 512]", alias_22: "f32[8, 8, 257, 64]", permute_273: "f32[1536, 512]", permute_280: "f32[512, 256]", permute_283: "f32[256, 1024]", permute_287: "f32[1024, 256]", div_21: "f32[8, 962, 1]", permute_291: "f32[256, 256]", alias_23: "f32[8, 4, 962, 64]", permute_297: "f32[768, 256]", div_22: "f32[8, 962, 1]", permute_301: "f32[256, 1024]", permute_305: "f32[1024, 256]", div_23: "f32[8, 962, 1]", permute_309: "f32[256, 256]", alias_24: "f32[8, 4, 962, 64]", permute_315: "f32[768, 256]", div_24: "f32[8, 962, 1]", permute_319: "f32[256, 1024]", permute_323: "f32[1024, 256]", div_25: "f32[8, 962, 1]", permute_327: "f32[256, 256]", alias_25: "f32[8, 4, 962, 64]", permute_333: "f32[768, 256]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(cat, getitem_1);  cat = getitem_1 = None
    mul: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_8: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(addmm_2, [8, 962, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(addmm_6, [8, 962, 1024]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_12: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_1: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_28: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(addmm_10, [8, 962, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_2: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_6: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(cat_1, getitem_49);  cat_1 = getitem_49 = None
    mul_21: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_14, [8, 257, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_26: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, 0.7071067811865476)
    erf_3: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_28: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_52: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_18, [8, 257, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_33: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476)
    erf_4: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_35: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_22, [8, 257, 2048]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476)
    erf_5: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_42: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_72: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_26, [8, 257, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_47: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476)
    erf_6: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_82: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_30, [8, 257, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_54: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476)
    erf_7: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_56: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_92: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(addmm_34, [8, 257, 2048]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_61: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476)
    erf_8: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_63: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_18: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(cat_2, getitem_145);  cat_2 = getitem_145 = None
    mul_63: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(addmm_38, [8, 65, 4096]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_9: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_71: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(addmm_42, [8, 65, 4096]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_75: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476)
    erf_10: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_78: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(addmm_46, [8, 65, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_82: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476)
    erf_11: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_85: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_136: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(addmm_50, [8, 65, 4096]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_89: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476)
    erf_12: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_92: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:283, code: x = self.head(x)
    mm_2: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_87);  permute_87 = None
    permute_88: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_3: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_88, clone_41);  permute_88 = clone_41 = None
    permute_89: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_140: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_90: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:280, code: x = x[:, 0]
    full_default: "f32[8, 1, 1024]" = torch.ops.aten.full.default([8, 1, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 1, 1024]" = torch.ops.aten.select_scatter.default(full_default, mm_2, 1, 0);  full_default = mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:260, code: cls_tokens = self.norm(cls_tokens)
    mul_94: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(select_scatter, primals_169);  primals_169 = None
    mul_95: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_94, 1024)
    sum_2: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True)
    mul_96: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_94, mul_91);  mul_94 = None
    sum_3: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True);  mul_96 = None
    mul_97: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_91, sum_3);  sum_3 = None
    sub_28: "f32[8, 1, 1024]" = torch.ops.aten.sub.Tensor(mul_95, sum_2);  mul_95 = sum_2 = None
    sub_29: "f32[8, 1, 1024]" = torch.ops.aten.sub.Tensor(sub_28, mul_97);  sub_28 = mul_97 = None
    mul_98: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(div, sub_29);  div = sub_29 = None
    mul_99: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(select_scatter, mul_91);  mul_91 = None
    sum_4: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_99, [0, 1]);  mul_99 = None
    sum_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(select_scatter, [0, 1]);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    full_default_2: "f32[8, 65, 1024]" = torch.ops.aten.full.default([8, 65, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[8, 65, 1024]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_98, 1, 0, 1);  full_default_2 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_141: "f32[520, 1024]" = torch.ops.aten.reshape.default(slice_scatter_1, [520, 1024])
    mm_4: "f32[520, 4096]" = torch.ops.aten.mm.default(view_141, permute_91);  permute_91 = None
    permute_92: "f32[1024, 520]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_5: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_92, view_137);  permute_92 = view_137 = None
    permute_93: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_6: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[1024]" = torch.ops.aten.reshape.default(sum_6, [1024]);  sum_6 = None
    permute_94: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    view_143: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(mm_4, [8, 65, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_101: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(add_92, 0.5);  add_92 = None
    mul_102: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, view_136)
    mul_103: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_102, -0.5);  mul_102 = None
    exp: "f32[8, 65, 4096]" = torch.ops.aten.exp.default(mul_103);  mul_103 = None
    mul_104: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_105: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, mul_104);  view_136 = mul_104 = None
    add_97: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(mul_101, mul_105);  mul_101 = mul_105 = None
    mul_106: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_143, add_97);  view_143 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[520, 4096]" = torch.ops.aten.reshape.default(mul_106, [520, 4096]);  mul_106 = None
    mm_6: "f32[520, 1024]" = torch.ops.aten.mm.default(view_144, permute_95);  permute_95 = None
    permute_96: "f32[4096, 520]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_7: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_96, view_135);  permute_96 = view_135 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[4096]" = torch.ops.aten.reshape.default(sum_7, [4096]);  sum_7 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    view_146: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_6, [8, 65, 1024]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_108: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_146, primals_163);  primals_163 = None
    mul_109: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_108, 1024)
    sum_8: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_108, mul_86);  mul_108 = None
    sum_9: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_86, sum_9);  sum_9 = None
    sub_31: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_109, sum_8);  mul_109 = sum_8 = None
    sub_32: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_31, mul_111);  sub_31 = mul_111 = None
    mul_112: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_1, sub_32);  div_1 = sub_32 = None
    mul_113: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_146, mul_86);  mul_86 = None
    sum_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_146, [0, 1]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_98: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(slice_scatter_1, mul_112);  slice_scatter_1 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_147: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_98, [520, 1024])
    mm_8: "f32[520, 1024]" = torch.ops.aten.mm.default(view_147, permute_99);  permute_99 = None
    permute_100: "f32[1024, 520]" = torch.ops.aten.permute.default(view_147, [1, 0])
    mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_100, view_133);  permute_100 = view_133 = None
    permute_101: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
    view_148: "f32[1024]" = torch.ops.aten.reshape.default(sum_12, [1024]);  sum_12 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    view_149: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_8, [8, 65, 1024]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_150: "f32[8, 65, 16, 64]" = torch.ops.aten.reshape.default(view_149, [8, 65, 16, 64]);  view_149 = None
    permute_103: "f32[8, 16, 65, 64]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_103, getitem_194, getitem_195, getitem_196, alias_13, getitem_198, getitem_199, getitem_200, 0, 0, 0.0, False, getitem_203, getitem_204);  permute_103 = getitem_194 = getitem_195 = getitem_196 = alias_13 = getitem_198 = getitem_199 = getitem_200 = getitem_203 = getitem_204 = None
    getitem_210: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_211: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_212: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[24, 16, 65, 64]" = torch.ops.aten.cat.default([getitem_210, getitem_211, getitem_212]);  getitem_210 = getitem_211 = getitem_212 = None
    view_151: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.reshape.default(cat_3, [3, 8, 16, 65, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_104: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.permute.default(view_151, [1, 3, 0, 2, 4]);  view_151 = None
    clone_43: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_152: "f32[8, 65, 3072]" = torch.ops.aten.reshape.default(clone_43, [8, 65, 3072]);  clone_43 = None
    view_153: "f32[520, 3072]" = torch.ops.aten.reshape.default(view_152, [520, 3072]);  view_152 = None
    mm_10: "f32[520, 1024]" = torch.ops.aten.mm.default(view_153, permute_105);  permute_105 = None
    permute_106: "f32[3072, 520]" = torch.ops.aten.permute.default(view_153, [1, 0])
    mm_11: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_106, view_129);  permute_106 = view_129 = None
    permute_107: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
    view_154: "f32[3072]" = torch.ops.aten.reshape.default(sum_13, [3072]);  sum_13 = None
    permute_108: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_155: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_10, [8, 65, 1024]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_115: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_155, primals_157);  primals_157 = None
    mul_116: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_115, 1024)
    sum_14: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [2], True)
    mul_117: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_115, mul_84);  mul_115 = None
    sum_15: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True);  mul_117 = None
    mul_118: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_84, sum_15);  sum_15 = None
    sub_34: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_116, sum_14);  mul_116 = sum_14 = None
    sub_35: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_34, mul_118);  sub_34 = mul_118 = None
    mul_119: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_2, sub_35);  div_2 = sub_35 = None
    mul_120: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_155, mul_84);  mul_84 = None
    sum_16: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_120, [0, 1]);  mul_120 = None
    sum_17: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_155, [0, 1]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_99: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_98, mul_119);  add_98 = mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_156: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_99, [520, 1024])
    mm_12: "f32[520, 4096]" = torch.ops.aten.mm.default(view_156, permute_109);  permute_109 = None
    permute_110: "f32[1024, 520]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_13: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_110, view_127);  permute_110 = view_127 = None
    permute_111: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_18: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
    view_157: "f32[1024]" = torch.ops.aten.reshape.default(sum_18, [1024]);  sum_18 = None
    permute_112: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_158: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(mm_12, [8, 65, 4096]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_122: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(add_85, 0.5);  add_85 = None
    mul_123: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, view_126)
    mul_124: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_123, -0.5);  mul_123 = None
    exp_1: "f32[8, 65, 4096]" = torch.ops.aten.exp.default(mul_124);  mul_124 = None
    mul_125: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_126: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, mul_125);  view_126 = mul_125 = None
    add_101: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(mul_122, mul_126);  mul_122 = mul_126 = None
    mul_127: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_158, add_101);  view_158 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_159: "f32[520, 4096]" = torch.ops.aten.reshape.default(mul_127, [520, 4096]);  mul_127 = None
    mm_14: "f32[520, 1024]" = torch.ops.aten.mm.default(view_159, permute_113);  permute_113 = None
    permute_114: "f32[4096, 520]" = torch.ops.aten.permute.default(view_159, [1, 0])
    mm_15: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_114, view_125);  permute_114 = view_125 = None
    permute_115: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_19: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
    view_160: "f32[4096]" = torch.ops.aten.reshape.default(sum_19, [4096]);  sum_19 = None
    permute_116: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    view_161: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_14, [8, 65, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_129: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_161, primals_151);  primals_151 = None
    mul_130: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_129, 1024)
    sum_20: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_129, mul_79);  mul_129 = None
    sum_21: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_79, sum_21);  sum_21 = None
    sub_37: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_130, sum_20);  mul_130 = sum_20 = None
    sub_38: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_37, mul_132);  sub_37 = mul_132 = None
    mul_133: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_3, sub_38);  div_3 = sub_38 = None
    mul_134: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_161, mul_79);  mul_79 = None
    sum_22: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_23: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_161, [0, 1]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_102: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_99, mul_133);  add_99 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_162: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_102, [520, 1024])
    mm_16: "f32[520, 1024]" = torch.ops.aten.mm.default(view_162, permute_117);  permute_117 = None
    permute_118: "f32[1024, 520]" = torch.ops.aten.permute.default(view_162, [1, 0])
    mm_17: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_118, view_123);  permute_118 = view_123 = None
    permute_119: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_24: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True);  view_162 = None
    view_163: "f32[1024]" = torch.ops.aten.reshape.default(sum_24, [1024]);  sum_24 = None
    permute_120: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    view_164: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_16, [8, 65, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_165: "f32[8, 65, 16, 64]" = torch.ops.aten.reshape.default(view_164, [8, 65, 16, 64]);  view_164 = None
    permute_121: "f32[8, 16, 65, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_121, getitem_178, getitem_179, getitem_180, alias_14, getitem_182, getitem_183, getitem_184, 0, 0, 0.0, False, getitem_187, getitem_188);  permute_121 = getitem_178 = getitem_179 = getitem_180 = alias_14 = getitem_182 = getitem_183 = getitem_184 = getitem_187 = getitem_188 = None
    getitem_213: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_214: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_215: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[24, 16, 65, 64]" = torch.ops.aten.cat.default([getitem_213, getitem_214, getitem_215]);  getitem_213 = getitem_214 = getitem_215 = None
    view_166: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.reshape.default(cat_4, [3, 8, 16, 65, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_122: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.permute.default(view_166, [1, 3, 0, 2, 4]);  view_166 = None
    clone_44: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    view_167: "f32[8, 65, 3072]" = torch.ops.aten.reshape.default(clone_44, [8, 65, 3072]);  clone_44 = None
    view_168: "f32[520, 3072]" = torch.ops.aten.reshape.default(view_167, [520, 3072]);  view_167 = None
    mm_18: "f32[520, 1024]" = torch.ops.aten.mm.default(view_168, permute_123);  permute_123 = None
    permute_124: "f32[3072, 520]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_19: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_124, view_119);  permute_124 = view_119 = None
    permute_125: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_25: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[3072]" = torch.ops.aten.reshape.default(sum_25, [3072]);  sum_25 = None
    permute_126: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_170: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_18, [8, 65, 1024]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_136: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_170, primals_145);  primals_145 = None
    mul_137: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_136, 1024)
    sum_26: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [2], True)
    mul_138: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_136, mul_77);  mul_136 = None
    sum_27: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
    mul_139: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_77, sum_27);  sum_27 = None
    sub_40: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_137, sum_26);  mul_137 = sum_26 = None
    sub_41: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_40, mul_139);  sub_40 = mul_139 = None
    mul_140: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_4, sub_41);  div_4 = sub_41 = None
    mul_141: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_170, mul_77);  mul_77 = None
    sum_28: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_141, [0, 1]);  mul_141 = None
    sum_29: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_103: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_102, mul_140);  add_102 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_171: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_103, [520, 1024])
    mm_20: "f32[520, 4096]" = torch.ops.aten.mm.default(view_171, permute_127);  permute_127 = None
    permute_128: "f32[1024, 520]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_21: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_128, view_117);  permute_128 = view_117 = None
    permute_129: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_30: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[1024]" = torch.ops.aten.reshape.default(sum_30, [1024]);  sum_30 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    view_173: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(mm_20, [8, 65, 4096]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_143: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_144: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, view_116)
    mul_145: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_144, -0.5);  mul_144 = None
    exp_2: "f32[8, 65, 4096]" = torch.ops.aten.exp.default(mul_145);  mul_145 = None
    mul_146: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_147: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, mul_146);  view_116 = mul_146 = None
    add_105: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(mul_143, mul_147);  mul_143 = mul_147 = None
    mul_148: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_173, add_105);  view_173 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_174: "f32[520, 4096]" = torch.ops.aten.reshape.default(mul_148, [520, 4096]);  mul_148 = None
    mm_22: "f32[520, 1024]" = torch.ops.aten.mm.default(view_174, permute_131);  permute_131 = None
    permute_132: "f32[4096, 520]" = torch.ops.aten.permute.default(view_174, [1, 0])
    mm_23: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_132, view_115);  permute_132 = view_115 = None
    permute_133: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
    view_175: "f32[4096]" = torch.ops.aten.reshape.default(sum_31, [4096]);  sum_31 = None
    permute_134: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
    view_176: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_22, [8, 65, 1024]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_150: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_176, primals_139);  primals_139 = None
    mul_151: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_150, 1024)
    sum_32: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_150, mul_72);  mul_150 = None
    sum_33: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_72, sum_33);  sum_33 = None
    sub_43: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_151, sum_32);  mul_151 = sum_32 = None
    sub_44: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_43, mul_153);  sub_43 = mul_153 = None
    mul_154: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_5, sub_44);  div_5 = sub_44 = None
    mul_155: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_176, mul_72);  mul_72 = None
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_176, [0, 1]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_106: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_103, mul_154);  add_103 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_177: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_106, [520, 1024])
    mm_24: "f32[520, 1024]" = torch.ops.aten.mm.default(view_177, permute_135);  permute_135 = None
    permute_136: "f32[1024, 520]" = torch.ops.aten.permute.default(view_177, [1, 0])
    mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_136, view_113);  permute_136 = view_113 = None
    permute_137: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_36: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
    view_178: "f32[1024]" = torch.ops.aten.reshape.default(sum_36, [1024]);  sum_36 = None
    permute_138: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_179: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_24, [8, 65, 1024]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_180: "f32[8, 65, 16, 64]" = torch.ops.aten.reshape.default(view_179, [8, 65, 16, 64]);  view_179 = None
    permute_139: "f32[8, 16, 65, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_139, getitem_162, getitem_163, getitem_164, alias_15, getitem_166, getitem_167, getitem_168, 0, 0, 0.0, False, getitem_171, getitem_172);  permute_139 = getitem_162 = getitem_163 = getitem_164 = alias_15 = getitem_166 = getitem_167 = getitem_168 = getitem_171 = getitem_172 = None
    getitem_216: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_2[0]
    getitem_217: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_2[1]
    getitem_218: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 16, 65, 64]" = torch.ops.aten.cat.default([getitem_216, getitem_217, getitem_218]);  getitem_216 = getitem_217 = getitem_218 = None
    view_181: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.reshape.default(cat_5, [3, 8, 16, 65, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_140: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.permute.default(view_181, [1, 3, 0, 2, 4]);  view_181 = None
    clone_45: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
    view_182: "f32[8, 65, 3072]" = torch.ops.aten.reshape.default(clone_45, [8, 65, 3072]);  clone_45 = None
    view_183: "f32[520, 3072]" = torch.ops.aten.reshape.default(view_182, [520, 3072]);  view_182 = None
    mm_26: "f32[520, 1024]" = torch.ops.aten.mm.default(view_183, permute_141);  permute_141 = None
    permute_142: "f32[3072, 520]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_27: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_142, view_109);  permute_142 = view_109 = None
    permute_143: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_37: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[3072]" = torch.ops.aten.reshape.default(sum_37, [3072]);  sum_37 = None
    permute_144: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_185: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_26, [8, 65, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_157: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_185, primals_133);  primals_133 = None
    mul_158: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_157, 1024)
    sum_38: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True)
    mul_159: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_157, mul_70);  mul_157 = None
    sum_39: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    mul_160: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_70, sum_39);  sum_39 = None
    sub_46: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_158, sum_38);  mul_158 = sum_38 = None
    sub_47: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_46, mul_160);  sub_46 = mul_160 = None
    mul_161: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_6, sub_47);  div_6 = sub_47 = None
    mul_162: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_185, mul_70);  mul_70 = None
    sum_40: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
    sum_41: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_107: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_106, mul_161);  add_106 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_186: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_107, [520, 1024])
    mm_28: "f32[520, 4096]" = torch.ops.aten.mm.default(view_186, permute_145);  permute_145 = None
    permute_146: "f32[1024, 520]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_29: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_146, view_107);  permute_146 = view_107 = None
    permute_147: "f32[4096, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_42: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[1024]" = torch.ops.aten.reshape.default(sum_42, [1024]);  sum_42 = None
    permute_148: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_188: "f32[8, 65, 4096]" = torch.ops.aten.reshape.default(mm_28, [8, 65, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_164: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
    mul_165: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_166: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_165, -0.5);  mul_165 = None
    exp_3: "f32[8, 65, 4096]" = torch.ops.aten.exp.default(mul_166);  mul_166 = None
    mul_167: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_168: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, mul_167);  view_106 = mul_167 = None
    add_109: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(mul_164, mul_168);  mul_164 = mul_168 = None
    mul_169: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_188, add_109);  view_188 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_189: "f32[520, 4096]" = torch.ops.aten.reshape.default(mul_169, [520, 4096]);  mul_169 = None
    mm_30: "f32[520, 1024]" = torch.ops.aten.mm.default(view_189, permute_149);  permute_149 = None
    permute_150: "f32[4096, 520]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_31: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_150, view_105);  permute_150 = view_105 = None
    permute_151: "f32[1024, 4096]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_43: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
    view_190: "f32[4096]" = torch.ops.aten.reshape.default(sum_43, [4096]);  sum_43 = None
    permute_152: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    view_191: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_30, [8, 65, 1024]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_171: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_191, primals_127);  primals_127 = None
    mul_172: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_171, 1024)
    sum_44: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_171, mul_65);  mul_171 = None
    sum_45: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_65, sum_45);  sum_45 = None
    sub_49: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_172, sum_44);  mul_172 = sum_44 = None
    sub_50: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_49, mul_174);  sub_49 = mul_174 = None
    mul_175: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_7, sub_50);  div_7 = sub_50 = None
    mul_176: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_191, mul_65);  mul_65 = None
    sum_46: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_47: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_110: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_107, mul_175);  add_107 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_192: "f32[520, 1024]" = torch.ops.aten.reshape.default(add_110, [520, 1024])
    mm_32: "f32[520, 1024]" = torch.ops.aten.mm.default(view_192, permute_153);  permute_153 = None
    permute_154: "f32[1024, 520]" = torch.ops.aten.permute.default(view_192, [1, 0])
    mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_154, view_103);  permute_154 = view_103 = None
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
    view_193: "f32[1024]" = torch.ops.aten.reshape.default(sum_48, [1024]);  sum_48 = None
    permute_156: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    view_194: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_32, [8, 65, 1024]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_195: "f32[8, 65, 16, 64]" = torch.ops.aten.reshape.default(view_194, [8, 65, 16, 64]);  view_194 = None
    permute_157: "f32[8, 16, 65, 64]" = torch.ops.aten.permute.default(view_195, [0, 2, 1, 3]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_157, getitem_146, getitem_147, getitem_148, alias_16, getitem_150, getitem_151, getitem_152, 0, 0, 0.0, False, getitem_155, getitem_156);  permute_157 = getitem_146 = getitem_147 = getitem_148 = alias_16 = getitem_150 = getitem_151 = getitem_152 = getitem_155 = getitem_156 = None
    getitem_219: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_3[0]
    getitem_220: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_3[1]
    getitem_221: "f32[8, 16, 65, 64]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 16, 65, 64]" = torch.ops.aten.cat.default([getitem_219, getitem_220, getitem_221]);  getitem_219 = getitem_220 = getitem_221 = None
    view_196: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.reshape.default(cat_6, [3, 8, 16, 65, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_158: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.permute.default(view_196, [1, 3, 0, 2, 4]);  view_196 = None
    clone_46: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    view_197: "f32[8, 65, 3072]" = torch.ops.aten.reshape.default(clone_46, [8, 65, 3072]);  clone_46 = None
    view_198: "f32[520, 3072]" = torch.ops.aten.reshape.default(view_197, [520, 3072]);  view_197 = None
    mm_34: "f32[520, 1024]" = torch.ops.aten.mm.default(view_198, permute_159);  permute_159 = None
    permute_160: "f32[3072, 520]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_35: "f32[3072, 1024]" = torch.ops.aten.mm.default(permute_160, view_99);  permute_160 = view_99 = None
    permute_161: "f32[1024, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_49: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[3072]" = torch.ops.aten.reshape.default(sum_49, [3072]);  sum_49 = None
    permute_162: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_200: "f32[8, 65, 1024]" = torch.ops.aten.reshape.default(mm_34, [8, 65, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_178: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_200, primals_121);  primals_121 = None
    mul_179: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_178, 1024)
    sum_50: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True)
    mul_180: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_178, mul_63);  mul_178 = None
    sum_51: "f32[8, 65, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True);  mul_180 = None
    mul_181: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_63, sum_51);  sum_51 = None
    sub_52: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(mul_179, sum_50);  mul_179 = sum_50 = None
    sub_53: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(sub_52, mul_181);  sub_52 = mul_181 = None
    div_8: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None
    mul_182: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(div_8, sub_53);  div_8 = sub_53 = None
    mul_183: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(view_200, mul_63);  mul_63 = None
    sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 1]);  mul_183 = None
    sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_200, [0, 1]);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_111: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_110, mul_182);  add_110 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_14: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(add_111, 1, 0, 1)
    slice_15: "f32[8, 64, 1024]" = torch.ops.aten.slice.Tensor(add_111, 1, 1, 65);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    permute_163: "f32[8, 1024, 64]" = torch.ops.aten.permute.default(slice_15, [0, 2, 1]);  slice_15 = None
    view_201: "f32[8, 1024, 8, 8]" = torch.ops.aten.reshape.default(permute_163, [8, 1024, 8, 8]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    sum_54: "f32[1, 1, 1024]" = torch.ops.aten.sum.dim_IntList(slice_14, [0, 1], True)
    view_202: "f32[1024]" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
    view_203: "f32[8, 1024]" = torch.ops.aten.reshape.default(slice_14, [8, 1024]);  slice_14 = None
    permute_164: "f32[1024, 8]" = torch.ops.aten.permute.default(view_203, [1, 0])
    mm_36: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_164, view_96);  permute_164 = view_96 = None
    permute_165: "f32[512, 1024]" = torch.ops.aten.permute.default(mm_36, [1, 0]);  mm_36 = None
    mm_37: "f32[8, 512]" = torch.ops.aten.mm.default(view_203, permute_166);  view_203 = permute_166 = None
    view_204: "f32[8, 1, 512]" = torch.ops.aten.reshape.default(mm_37, [8, 1, 512]);  mm_37 = None
    permute_167: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_201, view_95, primals_117, [1024], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True]);  view_201 = view_95 = primals_117 = None
    getitem_222: "f32[8, 512, 16, 16]" = convolution_backward[0]
    getitem_223: "f32[1024, 1, 3, 3]" = convolution_backward[1]
    getitem_224: "f32[1024]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    view_205: "f32[8, 512, 256]" = torch.ops.aten.reshape.default(getitem_222, [8, 512, 256]);  getitem_222 = None
    permute_168: "f32[8, 256, 512]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    full_default_4: "f32[8, 257, 512]" = torch.ops.aten.full.default([8, 257, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_3: "f32[8, 257, 512]" = torch.ops.aten.slice_scatter.default(full_default_4, permute_168, 1, 1, 9223372036854775807);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_scatter_5: "f32[8, 257, 512]" = torch.ops.aten.slice_scatter.default(full_default_4, view_204, 1, 0, 1);  full_default_4 = view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    add_112: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(slice_scatter_3, slice_scatter_5);  slice_scatter_3 = slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_206: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_112, [2056, 512])
    mm_38: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_206, permute_169);  permute_169 = None
    permute_170: "f32[512, 2056]" = torch.ops.aten.permute.default(view_206, [1, 0])
    mm_39: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_170, view_93);  permute_170 = view_93 = None
    permute_171: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_55: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_206, [0], True);  view_206 = None
    view_207: "f32[512]" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None
    permute_172: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    view_208: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_38, [8, 257, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_185: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_63, 0.5);  add_63 = None
    mul_186: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, view_92)
    mul_187: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_186, -0.5);  mul_186 = None
    exp_4: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_187);  mul_187 = None
    mul_188: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_189: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, mul_188);  view_92 = mul_188 = None
    add_114: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_185, mul_189);  mul_185 = mul_189 = None
    mul_190: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_208, add_114);  view_208 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_209: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_190, [2056, 2048]);  mul_190 = None
    mm_40: "f32[2056, 512]" = torch.ops.aten.mm.default(view_209, permute_173);  permute_173 = None
    permute_174: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_41: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_174, view_91);  permute_174 = view_91 = None
    permute_175: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_56: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[2048]" = torch.ops.aten.reshape.default(sum_56, [2048]);  sum_56 = None
    permute_176: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    view_211: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_40, [8, 257, 512]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_192: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_211, primals_111);  primals_111 = None
    mul_193: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_192, 512)
    sum_57: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_192, mul_58);  mul_192 = None
    sum_58: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_58, sum_58);  sum_58 = None
    sub_55: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_193, sum_57);  mul_193 = sum_57 = None
    sub_56: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_55, mul_195);  sub_55 = mul_195 = None
    mul_196: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_9, sub_56);  div_9 = sub_56 = None
    mul_197: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_211, mul_58);  mul_58 = None
    sum_59: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_60: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_115: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_112, mul_196);  add_112 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_212: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_115, [2056, 512])
    mm_42: "f32[2056, 512]" = torch.ops.aten.mm.default(view_212, permute_177);  permute_177 = None
    permute_178: "f32[512, 2056]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_43: "f32[512, 512]" = torch.ops.aten.mm.default(permute_178, view_89);  permute_178 = view_89 = None
    permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_61: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[512]" = torch.ops.aten.reshape.default(sum_61, [512]);  sum_61 = None
    permute_180: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    view_214: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_42, [8, 257, 512]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_215: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_214, [8, 257, 8, 64]);  view_214 = None
    permute_181: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_181, getitem_130, getitem_131, getitem_132, alias_17, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140);  permute_181 = getitem_130 = getitem_131 = getitem_132 = alias_17 = getitem_134 = getitem_135 = getitem_136 = getitem_139 = getitem_140 = None
    getitem_225: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_4[0]
    getitem_226: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_4[1]
    getitem_227: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_225, getitem_226, getitem_227]);  getitem_225 = getitem_226 = getitem_227 = None
    view_216: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_7, [3, 8, 8, 257, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_182: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_216, [1, 3, 0, 2, 4]);  view_216 = None
    clone_47: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    view_217: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_47, [8, 257, 1536]);  clone_47 = None
    view_218: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_217, [2056, 1536]);  view_217 = None
    mm_44: "f32[2056, 512]" = torch.ops.aten.mm.default(view_218, permute_183);  permute_183 = None
    permute_184: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_45: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_184, view_85);  permute_184 = view_85 = None
    permute_185: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_62: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[1536]" = torch.ops.aten.reshape.default(sum_62, [1536]);  sum_62 = None
    permute_186: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_220: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_44, [8, 257, 512]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_199: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_220, primals_105);  primals_105 = None
    mul_200: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_199, 512)
    sum_63: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True)
    mul_201: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_199, mul_56);  mul_199 = None
    sum_64: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
    mul_202: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_56, sum_64);  sum_64 = None
    sub_58: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_200, sum_63);  mul_200 = sum_63 = None
    sub_59: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_58, mul_202);  sub_58 = mul_202 = None
    mul_203: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_10, sub_59);  div_10 = sub_59 = None
    mul_204: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_220, mul_56);  mul_56 = None
    sum_65: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1]);  mul_204 = None
    sum_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_220, [0, 1]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_116: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_115, mul_203);  add_115 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_221: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_116, [2056, 512])
    mm_46: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_221, permute_187);  permute_187 = None
    permute_188: "f32[512, 2056]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_47: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_188, view_83);  permute_188 = view_83 = None
    permute_189: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_67: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
    view_222: "f32[512]" = torch.ops.aten.reshape.default(sum_67, [512]);  sum_67 = None
    permute_190: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_223: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_46, [8, 257, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_206: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_207: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, view_82)
    mul_208: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
    exp_5: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
    mul_209: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_210: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, mul_209);  view_82 = mul_209 = None
    add_118: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
    mul_211: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_223, add_118);  view_223 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_224: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_211, [2056, 2048]);  mul_211 = None
    mm_48: "f32[2056, 512]" = torch.ops.aten.mm.default(view_224, permute_191);  permute_191 = None
    permute_192: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_49: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_192, view_81);  permute_192 = view_81 = None
    permute_193: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_68: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[2048]" = torch.ops.aten.reshape.default(sum_68, [2048]);  sum_68 = None
    permute_194: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_226: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_48, [8, 257, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_213: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_226, primals_99);  primals_99 = None
    mul_214: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_213, 512)
    sum_69: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_213, mul_51);  mul_213 = None
    sum_70: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_51, sum_70);  sum_70 = None
    sub_61: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_214, sum_69);  mul_214 = sum_69 = None
    sub_62: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_61, mul_216);  sub_61 = mul_216 = None
    mul_217: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_62);  div_11 = sub_62 = None
    mul_218: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_226, mul_51);  mul_51 = None
    sum_71: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_72: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_119: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_116, mul_217);  add_116 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_227: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_119, [2056, 512])
    mm_50: "f32[2056, 512]" = torch.ops.aten.mm.default(view_227, permute_195);  permute_195 = None
    permute_196: "f32[512, 2056]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_51: "f32[512, 512]" = torch.ops.aten.mm.default(permute_196, view_79);  permute_196 = view_79 = None
    permute_197: "f32[512, 512]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_73: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[512]" = torch.ops.aten.reshape.default(sum_73, [512]);  sum_73 = None
    permute_198: "f32[512, 512]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_229: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_50, [8, 257, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_230: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_229, [8, 257, 8, 64]);  view_229 = None
    permute_199: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_199, getitem_114, getitem_115, getitem_116, alias_18, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124);  permute_199 = getitem_114 = getitem_115 = getitem_116 = alias_18 = getitem_118 = getitem_119 = getitem_120 = getitem_123 = getitem_124 = None
    getitem_228: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_5[0]
    getitem_229: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_5[1]
    getitem_230: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_228, getitem_229, getitem_230]);  getitem_228 = getitem_229 = getitem_230 = None
    view_231: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_8, [3, 8, 8, 257, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_200: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_231, [1, 3, 0, 2, 4]);  view_231 = None
    clone_48: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    view_232: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_48, [8, 257, 1536]);  clone_48 = None
    view_233: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_232, [2056, 1536]);  view_232 = None
    mm_52: "f32[2056, 512]" = torch.ops.aten.mm.default(view_233, permute_201);  permute_201 = None
    permute_202: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_53: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_202, view_75);  permute_202 = view_75 = None
    permute_203: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_74: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[1536]" = torch.ops.aten.reshape.default(sum_74, [1536]);  sum_74 = None
    permute_204: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    view_235: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_52, [8, 257, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_220: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_235, primals_93);  primals_93 = None
    mul_221: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_220, 512)
    sum_75: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
    mul_222: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_220, mul_49);  mul_220 = None
    sum_76: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_49, sum_76);  sum_76 = None
    sub_64: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_221, sum_75);  mul_221 = sum_75 = None
    sub_65: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_64, mul_223);  sub_64 = mul_223 = None
    mul_224: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_12, sub_65);  div_12 = sub_65 = None
    mul_225: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_235, mul_49);  mul_49 = None
    sum_77: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
    sum_78: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_235, [0, 1]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_120: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_119, mul_224);  add_119 = mul_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_236: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_120, [2056, 512])
    mm_54: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_236, permute_205);  permute_205 = None
    permute_206: "f32[512, 2056]" = torch.ops.aten.permute.default(view_236, [1, 0])
    mm_55: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_206, view_73);  permute_206 = view_73 = None
    permute_207: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_79: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True);  view_236 = None
    view_237: "f32[512]" = torch.ops.aten.reshape.default(sum_79, [512]);  sum_79 = None
    permute_208: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_238: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_54, [8, 257, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_227: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_228: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, view_72)
    mul_229: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_228, -0.5);  mul_228 = None
    exp_6: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_229);  mul_229 = None
    mul_230: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_231: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, mul_230);  view_72 = mul_230 = None
    add_122: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_227, mul_231);  mul_227 = mul_231 = None
    mul_232: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_238, add_122);  view_238 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_239: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_232, [2056, 2048]);  mul_232 = None
    mm_56: "f32[2056, 512]" = torch.ops.aten.mm.default(view_239, permute_209);  permute_209 = None
    permute_210: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_57: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_210, view_71);  permute_210 = view_71 = None
    permute_211: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_80: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[2048]" = torch.ops.aten.reshape.default(sum_80, [2048]);  sum_80 = None
    permute_212: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_241: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_56, [8, 257, 512]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_234: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_241, primals_87);  primals_87 = None
    mul_235: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_234, 512)
    sum_81: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_234, mul_44);  mul_234 = None
    sum_82: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_44, sum_82);  sum_82 = None
    sub_67: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_235, sum_81);  mul_235 = sum_81 = None
    sub_68: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_67, mul_237);  sub_67 = mul_237 = None
    mul_238: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_13, sub_68);  div_13 = sub_68 = None
    mul_239: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_241, mul_44);  mul_44 = None
    sum_83: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_84: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_120, mul_238);  add_120 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_242: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_123, [2056, 512])
    mm_58: "f32[2056, 512]" = torch.ops.aten.mm.default(view_242, permute_213);  permute_213 = None
    permute_214: "f32[512, 2056]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_59: "f32[512, 512]" = torch.ops.aten.mm.default(permute_214, view_69);  permute_214 = view_69 = None
    permute_215: "f32[512, 512]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_85: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[512]" = torch.ops.aten.reshape.default(sum_85, [512]);  sum_85 = None
    permute_216: "f32[512, 512]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_244: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_58, [8, 257, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_245: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_244, [8, 257, 8, 64]);  view_244 = None
    permute_217: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_245, [0, 2, 1, 3]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_217, getitem_98, getitem_99, getitem_100, alias_19, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108);  permute_217 = getitem_98 = getitem_99 = getitem_100 = alias_19 = getitem_102 = getitem_103 = getitem_104 = getitem_107 = getitem_108 = None
    getitem_231: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_6[0]
    getitem_232: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_6[1]
    getitem_233: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_231, getitem_232, getitem_233]);  getitem_231 = getitem_232 = getitem_233 = None
    view_246: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_9, [3, 8, 8, 257, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_218: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_246, [1, 3, 0, 2, 4]);  view_246 = None
    clone_49: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_247: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_49, [8, 257, 1536]);  clone_49 = None
    view_248: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_247, [2056, 1536]);  view_247 = None
    mm_60: "f32[2056, 512]" = torch.ops.aten.mm.default(view_248, permute_219);  permute_219 = None
    permute_220: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_61: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_220, view_65);  permute_220 = view_65 = None
    permute_221: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_86: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[1536]" = torch.ops.aten.reshape.default(sum_86, [1536]);  sum_86 = None
    permute_222: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    view_250: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_60, [8, 257, 512]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_241: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_250, primals_81);  primals_81 = None
    mul_242: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_241, 512)
    sum_87: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_241, mul_42);  mul_241 = None
    sum_88: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_42, sum_88);  sum_88 = None
    sub_70: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_242, sum_87);  mul_242 = sum_87 = None
    sub_71: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_70, mul_244);  sub_70 = mul_244 = None
    mul_245: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_14, sub_71);  div_14 = sub_71 = None
    mul_246: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_250, mul_42);  mul_42 = None
    sum_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_90: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_250, [0, 1]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_124: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_123, mul_245);  add_123 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_251: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_124, [2056, 512])
    mm_62: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_251, permute_223);  permute_223 = None
    permute_224: "f32[512, 2056]" = torch.ops.aten.permute.default(view_251, [1, 0])
    mm_63: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_224, view_63);  permute_224 = view_63 = None
    permute_225: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_91: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_251, [0], True);  view_251 = None
    view_252: "f32[512]" = torch.ops.aten.reshape.default(sum_91, [512]);  sum_91 = None
    permute_226: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_253: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_62, [8, 257, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_248: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
    mul_249: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, view_62)
    mul_250: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_249, -0.5);  mul_249 = None
    exp_7: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_250);  mul_250 = None
    mul_251: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_252: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, mul_251);  view_62 = mul_251 = None
    add_126: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_248, mul_252);  mul_248 = mul_252 = None
    mul_253: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_253, add_126);  view_253 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_254: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_253, [2056, 2048]);  mul_253 = None
    mm_64: "f32[2056, 512]" = torch.ops.aten.mm.default(view_254, permute_227);  permute_227 = None
    permute_228: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_65: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_228, view_61);  permute_228 = view_61 = None
    permute_229: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_92: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[2048]" = torch.ops.aten.reshape.default(sum_92, [2048]);  sum_92 = None
    permute_230: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_256: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_64, [8, 257, 512]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_255: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_256, primals_75);  primals_75 = None
    mul_256: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_255, 512)
    sum_93: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_255, mul_37);  mul_255 = None
    sum_94: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_37, sum_94);  sum_94 = None
    sub_73: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_256, sum_93);  mul_256 = sum_93 = None
    sub_74: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_73, mul_258);  sub_73 = mul_258 = None
    mul_259: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_15, sub_74);  div_15 = sub_74 = None
    mul_260: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_256, mul_37);  mul_37 = None
    sum_95: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_96: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_127: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_124, mul_259);  add_124 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_257: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_127, [2056, 512])
    mm_66: "f32[2056, 512]" = torch.ops.aten.mm.default(view_257, permute_231);  permute_231 = None
    permute_232: "f32[512, 2056]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_67: "f32[512, 512]" = torch.ops.aten.mm.default(permute_232, view_59);  permute_232 = view_59 = None
    permute_233: "f32[512, 512]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_97: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[512]" = torch.ops.aten.reshape.default(sum_97, [512]);  sum_97 = None
    permute_234: "f32[512, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_259: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_66, [8, 257, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_260: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_259, [8, 257, 8, 64]);  view_259 = None
    permute_235: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_260, [0, 2, 1, 3]);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_235, getitem_82, getitem_83, getitem_84, alias_20, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92);  permute_235 = getitem_82 = getitem_83 = getitem_84 = alias_20 = getitem_86 = getitem_87 = getitem_88 = getitem_91 = getitem_92 = None
    getitem_234: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_7[0]
    getitem_235: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_7[1]
    getitem_236: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_234, getitem_235, getitem_236]);  getitem_234 = getitem_235 = getitem_236 = None
    view_261: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_10, [3, 8, 8, 257, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_236: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_261, [1, 3, 0, 2, 4]);  view_261 = None
    clone_50: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    view_262: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_50, [8, 257, 1536]);  clone_50 = None
    view_263: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_262, [2056, 1536]);  view_262 = None
    mm_68: "f32[2056, 512]" = torch.ops.aten.mm.default(view_263, permute_237);  permute_237 = None
    permute_238: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_69: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_238, view_55);  permute_238 = view_55 = None
    permute_239: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_98: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[1536]" = torch.ops.aten.reshape.default(sum_98, [1536]);  sum_98 = None
    permute_240: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_265: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_68, [8, 257, 512]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_262: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_265, primals_69);  primals_69 = None
    mul_263: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_262, 512)
    sum_99: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True)
    mul_264: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_262, mul_35);  mul_262 = None
    sum_100: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
    mul_265: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_35, sum_100);  sum_100 = None
    sub_76: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_263, sum_99);  mul_263 = sum_99 = None
    sub_77: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_76, mul_265);  sub_76 = mul_265 = None
    mul_266: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_16, sub_77);  div_16 = sub_77 = None
    mul_267: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_265, mul_35);  mul_35 = None
    sum_101: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1]);  mul_267 = None
    sum_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_128: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_127, mul_266);  add_127 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_266: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_128, [2056, 512])
    mm_70: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_266, permute_241);  permute_241 = None
    permute_242: "f32[512, 2056]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_71: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_242, view_53);  permute_242 = view_53 = None
    permute_243: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_103: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[512]" = torch.ops.aten.reshape.default(sum_103, [512]);  sum_103 = None
    permute_244: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_268: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_70, [8, 257, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_269: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
    mul_270: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, view_52)
    mul_271: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_270, -0.5);  mul_270 = None
    exp_8: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_271);  mul_271 = None
    mul_272: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_273: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, mul_272);  view_52 = mul_272 = None
    add_130: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_269, mul_273);  mul_269 = mul_273 = None
    mul_274: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_268, add_130);  view_268 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_269: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_274, [2056, 2048]);  mul_274 = None
    mm_72: "f32[2056, 512]" = torch.ops.aten.mm.default(view_269, permute_245);  permute_245 = None
    permute_246: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_73: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_246, view_51);  permute_246 = view_51 = None
    permute_247: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_104: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[2048]" = torch.ops.aten.reshape.default(sum_104, [2048]);  sum_104 = None
    permute_248: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_271: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_72, [8, 257, 512]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_276: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_271, primals_63);  primals_63 = None
    mul_277: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_276, 512)
    sum_105: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
    mul_278: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_276, mul_30);  mul_276 = None
    sum_106: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
    mul_279: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_30, sum_106);  sum_106 = None
    sub_79: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_277, sum_105);  mul_277 = sum_105 = None
    sub_80: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_79, mul_279);  sub_79 = mul_279 = None
    mul_280: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_17, sub_80);  div_17 = sub_80 = None
    mul_281: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_271, mul_30);  mul_30 = None
    sum_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
    sum_108: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_131: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_128, mul_280);  add_128 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_272: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_131, [2056, 512])
    mm_74: "f32[2056, 512]" = torch.ops.aten.mm.default(view_272, permute_249);  permute_249 = None
    permute_250: "f32[512, 2056]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_75: "f32[512, 512]" = torch.ops.aten.mm.default(permute_250, view_49);  permute_250 = view_49 = None
    permute_251: "f32[512, 512]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_109: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[512]" = torch.ops.aten.reshape.default(sum_109, [512]);  sum_109 = None
    permute_252: "f32[512, 512]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_274: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_74, [8, 257, 512]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_275: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_274, [8, 257, 8, 64]);  view_274 = None
    permute_253: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_253, getitem_66, getitem_67, getitem_68, alias_21, getitem_70, getitem_71, getitem_72, 0, 0, 0.0, False, getitem_75, getitem_76);  permute_253 = getitem_66 = getitem_67 = getitem_68 = alias_21 = getitem_70 = getitem_71 = getitem_72 = getitem_75 = getitem_76 = None
    getitem_237: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_8[0]
    getitem_238: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_8[1]
    getitem_239: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_237, getitem_238, getitem_239]);  getitem_237 = getitem_238 = getitem_239 = None
    view_276: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_11, [3, 8, 8, 257, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_254: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_276, [1, 3, 0, 2, 4]);  view_276 = None
    clone_51: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
    view_277: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_51, [8, 257, 1536]);  clone_51 = None
    view_278: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_277, [2056, 1536]);  view_277 = None
    mm_76: "f32[2056, 512]" = torch.ops.aten.mm.default(view_278, permute_255);  permute_255 = None
    permute_256: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_77: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_256, view_45);  permute_256 = view_45 = None
    permute_257: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_110: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[1536]" = torch.ops.aten.reshape.default(sum_110, [1536]);  sum_110 = None
    permute_258: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_280: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_76, [8, 257, 512]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_283: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_280, primals_57);  primals_57 = None
    mul_284: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_283, 512)
    sum_111: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_283, mul_28);  mul_283 = None
    sum_112: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_28, sum_112);  sum_112 = None
    sub_82: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_284, sum_111);  mul_284 = sum_111 = None
    sub_83: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_82, mul_286);  sub_82 = mul_286 = None
    mul_287: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_18, sub_83);  div_18 = sub_83 = None
    mul_288: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_280, mul_28);  mul_28 = None
    sum_113: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_114: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_280, [0, 1]);  view_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_132: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_131, mul_287);  add_131 = mul_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_281: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_132, [2056, 512])
    mm_78: "f32[2056, 2048]" = torch.ops.aten.mm.default(view_281, permute_259);  permute_259 = None
    permute_260: "f32[512, 2056]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_79: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_260, view_43);  permute_260 = view_43 = None
    permute_261: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_115: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[512]" = torch.ops.aten.reshape.default(sum_115, [512]);  sum_115 = None
    permute_262: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    view_283: "f32[8, 257, 2048]" = torch.ops.aten.reshape.default(mm_78, [8, 257, 2048]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_290: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(add_28, 0.5);  add_28 = None
    mul_291: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, view_42)
    mul_292: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_291, -0.5);  mul_291 = None
    exp_9: "f32[8, 257, 2048]" = torch.ops.aten.exp.default(mul_292);  mul_292 = None
    mul_293: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_294: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, mul_293);  view_42 = mul_293 = None
    add_134: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(mul_290, mul_294);  mul_290 = mul_294 = None
    mul_295: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_283, add_134);  view_283 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_284: "f32[2056, 2048]" = torch.ops.aten.reshape.default(mul_295, [2056, 2048]);  mul_295 = None
    mm_80: "f32[2056, 512]" = torch.ops.aten.mm.default(view_284, permute_263);  permute_263 = None
    permute_264: "f32[2048, 2056]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_81: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_264, view_41);  permute_264 = view_41 = None
    permute_265: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_116: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[2048]" = torch.ops.aten.reshape.default(sum_116, [2048]);  sum_116 = None
    permute_266: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_286: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_80, [8, 257, 512]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_297: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_286, primals_51);  primals_51 = None
    mul_298: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_297, 512)
    sum_117: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_297, mul_23);  mul_297 = None
    sum_118: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_23, sum_118);  sum_118 = None
    sub_85: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_298, sum_117);  mul_298 = sum_117 = None
    sub_86: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_85, mul_300);  sub_85 = mul_300 = None
    mul_301: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_19, sub_86);  div_19 = sub_86 = None
    mul_302: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_286, mul_23);  mul_23 = None
    sum_119: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_286, [0, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_135: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_132, mul_301);  add_132 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_287: "f32[2056, 512]" = torch.ops.aten.reshape.default(add_135, [2056, 512])
    mm_82: "f32[2056, 512]" = torch.ops.aten.mm.default(view_287, permute_267);  permute_267 = None
    permute_268: "f32[512, 2056]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_83: "f32[512, 512]" = torch.ops.aten.mm.default(permute_268, view_39);  permute_268 = view_39 = None
    permute_269: "f32[512, 512]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[512]" = torch.ops.aten.reshape.default(sum_121, [512]);  sum_121 = None
    permute_270: "f32[512, 512]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_289: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_82, [8, 257, 512]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_290: "f32[8, 257, 8, 64]" = torch.ops.aten.reshape.default(view_289, [8, 257, 8, 64]);  view_289 = None
    permute_271: "f32[8, 8, 257, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_271, getitem_50, getitem_51, getitem_52, alias_22, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60);  permute_271 = getitem_50 = getitem_51 = getitem_52 = alias_22 = getitem_54 = getitem_55 = getitem_56 = getitem_59 = getitem_60 = None
    getitem_240: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_9[0]
    getitem_241: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_9[1]
    getitem_242: "f32[8, 8, 257, 64]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 8, 257, 64]" = torch.ops.aten.cat.default([getitem_240, getitem_241, getitem_242]);  getitem_240 = getitem_241 = getitem_242 = None
    view_291: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.reshape.default(cat_12, [3, 8, 8, 257, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_272: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.permute.default(view_291, [1, 3, 0, 2, 4]);  view_291 = None
    clone_52: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.clone.default(permute_272, memory_format = torch.contiguous_format);  permute_272 = None
    view_292: "f32[8, 257, 1536]" = torch.ops.aten.reshape.default(clone_52, [8, 257, 1536]);  clone_52 = None
    view_293: "f32[2056, 1536]" = torch.ops.aten.reshape.default(view_292, [2056, 1536]);  view_292 = None
    mm_84: "f32[2056, 512]" = torch.ops.aten.mm.default(view_293, permute_273);  permute_273 = None
    permute_274: "f32[1536, 2056]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_85: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_274, view_35);  permute_274 = view_35 = None
    permute_275: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[1536]" = torch.ops.aten.reshape.default(sum_122, [1536]);  sum_122 = None
    permute_276: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_295: "f32[8, 257, 512]" = torch.ops.aten.reshape.default(mm_84, [8, 257, 512]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_304: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_295, primals_45);  primals_45 = None
    mul_305: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_304, 512)
    sum_123: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
    mul_306: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_304, mul_21);  mul_304 = None
    sum_124: "f32[8, 257, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
    mul_307: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_21, sum_124);  sum_124 = None
    sub_88: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(mul_305, sum_123);  mul_305 = sum_123 = None
    sub_89: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(sub_88, mul_307);  sub_88 = mul_307 = None
    div_20: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 512);  rsqrt_6 = None
    mul_308: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(div_20, sub_89);  div_20 = sub_89 = None
    mul_309: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(view_295, mul_21);  mul_21 = None
    sum_125: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
    sum_126: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_136: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_135, mul_308);  add_135 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_16: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_136, 1, 0, 1)
    slice_17: "f32[8, 256, 512]" = torch.ops.aten.slice.Tensor(add_136, 1, 1, 257);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    permute_277: "f32[8, 512, 256]" = torch.ops.aten.permute.default(slice_17, [0, 2, 1]);  slice_17 = None
    view_296: "f32[8, 512, 16, 16]" = torch.ops.aten.reshape.default(permute_277, [8, 512, 16, 16]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    sum_127: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(slice_16, [0, 1], True)
    view_297: "f32[512]" = torch.ops.aten.reshape.default(sum_127, [512]);  sum_127 = None
    view_298: "f32[8, 512]" = torch.ops.aten.reshape.default(slice_16, [8, 512]);  slice_16 = None
    permute_278: "f32[512, 8]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_86: "f32[512, 256]" = torch.ops.aten.mm.default(permute_278, view_32);  permute_278 = view_32 = None
    permute_279: "f32[256, 512]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    mm_87: "f32[8, 256]" = torch.ops.aten.mm.default(view_298, permute_280);  view_298 = permute_280 = None
    view_299: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mm_87, [8, 1, 256]);  mm_87 = None
    permute_281: "f32[512, 256]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_296, view_31, primals_41, [512], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, True]);  view_296 = view_31 = primals_41 = None
    getitem_243: "f32[8, 256, 31, 31]" = convolution_backward_1[0]
    getitem_244: "f32[512, 1, 3, 3]" = convolution_backward_1[1]
    getitem_245: "f32[512]" = convolution_backward_1[2];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    view_300: "f32[8, 256, 961]" = torch.ops.aten.reshape.default(getitem_243, [8, 256, 961]);  getitem_243 = None
    permute_282: "f32[8, 961, 256]" = torch.ops.aten.permute.default(view_300, [0, 2, 1]);  view_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    full_default_8: "f32[8, 962, 256]" = torch.ops.aten.full.default([8, 962, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[8, 962, 256]" = torch.ops.aten.slice_scatter.default(full_default_8, permute_282, 1, 1, 9223372036854775807);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_scatter_9: "f32[8, 962, 256]" = torch.ops.aten.slice_scatter.default(full_default_8, view_299, 1, 0, 1);  full_default_8 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    add_137: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(slice_scatter_7, slice_scatter_9);  slice_scatter_7 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_301: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_137, [7696, 256])
    mm_88: "f32[7696, 1024]" = torch.ops.aten.mm.default(view_301, permute_283);  permute_283 = None
    permute_284: "f32[256, 7696]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_89: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_284, view_29);  permute_284 = view_29 = None
    permute_285: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_128: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[256]" = torch.ops.aten.reshape.default(sum_128, [256]);  sum_128 = None
    permute_286: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_303: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(mm_88, [8, 962, 1024]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_311: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(add_20, 0.5);  add_20 = None
    mul_312: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, view_28)
    mul_313: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_312, -0.5);  mul_312 = None
    exp_10: "f32[8, 962, 1024]" = torch.ops.aten.exp.default(mul_313);  mul_313 = None
    mul_314: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_315: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, mul_314);  view_28 = mul_314 = None
    add_139: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(mul_311, mul_315);  mul_311 = mul_315 = None
    mul_316: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_303, add_139);  view_303 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_304: "f32[7696, 1024]" = torch.ops.aten.reshape.default(mul_316, [7696, 1024]);  mul_316 = None
    mm_90: "f32[7696, 256]" = torch.ops.aten.mm.default(view_304, permute_287);  permute_287 = None
    permute_288: "f32[1024, 7696]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_91: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_288, view_27);  permute_288 = view_27 = None
    permute_289: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_129: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[1024]" = torch.ops.aten.reshape.default(sum_129, [1024]);  sum_129 = None
    permute_290: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    view_306: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_90, [8, 962, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_318: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_306, primals_35);  primals_35 = None
    mul_319: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_318, 256)
    sum_130: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_318, mul_16);  mul_318 = None
    sum_131: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_16, sum_131);  sum_131 = None
    sub_91: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_319, sum_130);  mul_319 = sum_130 = None
    sub_92: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_91, mul_321);  sub_91 = mul_321 = None
    mul_322: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_21, sub_92);  div_21 = sub_92 = None
    mul_323: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_306, mul_16);  mul_16 = None
    sum_132: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_133: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_306, [0, 1]);  view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_140: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_137, mul_322);  add_137 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_307: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_140, [7696, 256])
    mm_92: "f32[7696, 256]" = torch.ops.aten.mm.default(view_307, permute_291);  permute_291 = None
    permute_292: "f32[256, 7696]" = torch.ops.aten.permute.default(view_307, [1, 0])
    mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_292, view_25);  permute_292 = view_25 = None
    permute_293: "f32[256, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_134: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
    view_308: "f32[256]" = torch.ops.aten.reshape.default(sum_134, [256]);  sum_134 = None
    permute_294: "f32[256, 256]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
    view_309: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_92, [8, 962, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_310: "f32[8, 962, 4, 64]" = torch.ops.aten.reshape.default(view_309, [8, 962, 4, 64]);  view_309 = None
    permute_295: "f32[8, 4, 962, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_295, getitem_34, getitem_35, getitem_36, alias_23, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44);  permute_295 = getitem_34 = getitem_35 = getitem_36 = alias_23 = getitem_38 = getitem_39 = getitem_40 = getitem_43 = getitem_44 = None
    getitem_246: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_10[0]
    getitem_247: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_10[1]
    getitem_248: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_13: "f32[24, 4, 962, 64]" = torch.ops.aten.cat.default([getitem_246, getitem_247, getitem_248]);  getitem_246 = getitem_247 = getitem_248 = None
    view_311: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.reshape.default(cat_13, [3, 8, 4, 962, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_296: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.permute.default(view_311, [1, 3, 0, 2, 4]);  view_311 = None
    clone_53: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.clone.default(permute_296, memory_format = torch.contiguous_format);  permute_296 = None
    view_312: "f32[8, 962, 768]" = torch.ops.aten.reshape.default(clone_53, [8, 962, 768]);  clone_53 = None
    view_313: "f32[7696, 768]" = torch.ops.aten.reshape.default(view_312, [7696, 768]);  view_312 = None
    mm_94: "f32[7696, 256]" = torch.ops.aten.mm.default(view_313, permute_297);  permute_297 = None
    permute_298: "f32[768, 7696]" = torch.ops.aten.permute.default(view_313, [1, 0])
    mm_95: "f32[768, 256]" = torch.ops.aten.mm.default(permute_298, view_21);  permute_298 = view_21 = None
    permute_299: "f32[256, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_135: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
    view_314: "f32[768]" = torch.ops.aten.reshape.default(sum_135, [768]);  sum_135 = None
    permute_300: "f32[768, 256]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    view_315: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_94, [8, 962, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_325: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_315, primals_29);  primals_29 = None
    mul_326: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_325, 256)
    sum_136: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_325, mul_14);  mul_325 = None
    sum_137: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_14, sum_137);  sum_137 = None
    sub_94: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_326, sum_136);  mul_326 = sum_136 = None
    sub_95: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_94, mul_328);  sub_94 = mul_328 = None
    mul_329: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_22, sub_95);  div_22 = sub_95 = None
    mul_330: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_315, mul_14);  mul_14 = None
    sum_138: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_139: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_315, [0, 1]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_141: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_140, mul_329);  add_140 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_316: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_141, [7696, 256])
    mm_96: "f32[7696, 1024]" = torch.ops.aten.mm.default(view_316, permute_301);  permute_301 = None
    permute_302: "f32[256, 7696]" = torch.ops.aten.permute.default(view_316, [1, 0])
    mm_97: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_302, view_19);  permute_302 = view_19 = None
    permute_303: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
    view_317: "f32[256]" = torch.ops.aten.reshape.default(sum_140, [256]);  sum_140 = None
    permute_304: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    view_318: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(mm_96, [8, 962, 1024]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_332: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(add_13, 0.5);  add_13 = None
    mul_333: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_334: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_333, -0.5);  mul_333 = None
    exp_11: "f32[8, 962, 1024]" = torch.ops.aten.exp.default(mul_334);  mul_334 = None
    mul_335: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_336: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, mul_335);  view_18 = mul_335 = None
    add_143: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
    mul_337: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_318, add_143);  view_318 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_319: "f32[7696, 1024]" = torch.ops.aten.reshape.default(mul_337, [7696, 1024]);  mul_337 = None
    mm_98: "f32[7696, 256]" = torch.ops.aten.mm.default(view_319, permute_305);  permute_305 = None
    permute_306: "f32[1024, 7696]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_99: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_306, view_17);  permute_306 = view_17 = None
    permute_307: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_141: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
    view_320: "f32[1024]" = torch.ops.aten.reshape.default(sum_141, [1024]);  sum_141 = None
    permute_308: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    view_321: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_98, [8, 962, 256]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_339: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_321, primals_23);  primals_23 = None
    mul_340: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_339, 256)
    sum_142: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_339, mul_9);  mul_339 = None
    sum_143: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_9, sum_143);  sum_143 = None
    sub_97: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_340, sum_142);  mul_340 = sum_142 = None
    sub_98: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_97, mul_342);  sub_97 = mul_342 = None
    mul_343: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_23, sub_98);  div_23 = sub_98 = None
    mul_344: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_321, mul_9);  mul_9 = None
    sum_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_321, [0, 1]);  view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_144: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_141, mul_343);  add_141 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_322: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_144, [7696, 256])
    mm_100: "f32[7696, 256]" = torch.ops.aten.mm.default(view_322, permute_309);  permute_309 = None
    permute_310: "f32[256, 7696]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_101: "f32[256, 256]" = torch.ops.aten.mm.default(permute_310, view_15);  permute_310 = view_15 = None
    permute_311: "f32[256, 256]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_146: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[256]" = torch.ops.aten.reshape.default(sum_146, [256]);  sum_146 = None
    permute_312: "f32[256, 256]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_324: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_100, [8, 962, 256]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_325: "f32[8, 962, 4, 64]" = torch.ops.aten.reshape.default(view_324, [8, 962, 4, 64]);  view_324 = None
    permute_313: "f32[8, 4, 962, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_313, getitem_18, getitem_19, getitem_20, alias_24, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28);  permute_313 = getitem_18 = getitem_19 = getitem_20 = alias_24 = getitem_22 = getitem_23 = getitem_24 = getitem_27 = getitem_28 = None
    getitem_249: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_11[0]
    getitem_250: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_11[1]
    getitem_251: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_14: "f32[24, 4, 962, 64]" = torch.ops.aten.cat.default([getitem_249, getitem_250, getitem_251]);  getitem_249 = getitem_250 = getitem_251 = None
    view_326: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.reshape.default(cat_14, [3, 8, 4, 962, 64]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_314: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.permute.default(view_326, [1, 3, 0, 2, 4]);  view_326 = None
    clone_54: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
    view_327: "f32[8, 962, 768]" = torch.ops.aten.reshape.default(clone_54, [8, 962, 768]);  clone_54 = None
    view_328: "f32[7696, 768]" = torch.ops.aten.reshape.default(view_327, [7696, 768]);  view_327 = None
    mm_102: "f32[7696, 256]" = torch.ops.aten.mm.default(view_328, permute_315);  permute_315 = None
    permute_316: "f32[768, 7696]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_103: "f32[768, 256]" = torch.ops.aten.mm.default(permute_316, view_11);  permute_316 = view_11 = None
    permute_317: "f32[256, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
    permute_318: "f32[768, 256]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_330: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_102, [8, 962, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_346: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_330, primals_17);  primals_17 = None
    mul_347: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_346, 256)
    sum_148: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
    mul_348: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_346, mul_7);  mul_346 = None
    sum_149: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
    mul_349: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_7, sum_149);  sum_149 = None
    sub_100: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_347, sum_148);  mul_347 = sum_148 = None
    sub_101: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_100, mul_349);  sub_100 = mul_349 = None
    mul_350: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_24, sub_101);  div_24 = sub_101 = None
    mul_351: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_330, mul_7);  mul_7 = None
    sum_150: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 1]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_145: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_144, mul_350);  add_144 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_331: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_145, [7696, 256])
    mm_104: "f32[7696, 1024]" = torch.ops.aten.mm.default(view_331, permute_319);  permute_319 = None
    permute_320: "f32[256, 7696]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_105: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_320, view_9);  permute_320 = view_9 = None
    permute_321: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_152: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[256]" = torch.ops.aten.reshape.default(sum_152, [256]);  sum_152 = None
    permute_322: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    view_333: "f32[8, 962, 1024]" = torch.ops.aten.reshape.default(mm_104, [8, 962, 1024]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_353: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(add_6, 0.5);  add_6 = None
    mul_354: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_355: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_354, -0.5);  mul_354 = None
    exp_12: "f32[8, 962, 1024]" = torch.ops.aten.exp.default(mul_355);  mul_355 = None
    mul_356: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_357: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, mul_356);  view_8 = mul_356 = None
    add_147: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(mul_353, mul_357);  mul_353 = mul_357 = None
    mul_358: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_333, add_147);  view_333 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_334: "f32[7696, 1024]" = torch.ops.aten.reshape.default(mul_358, [7696, 1024]);  mul_358 = None
    mm_106: "f32[7696, 256]" = torch.ops.aten.mm.default(view_334, permute_323);  permute_323 = None
    permute_324: "f32[1024, 7696]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_107: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_324, view_7);  permute_324 = view_7 = None
    permute_325: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[1024]" = torch.ops.aten.reshape.default(sum_153, [1024]);  sum_153 = None
    permute_326: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    view_336: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_106, [8, 962, 256]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    mul_360: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_336, primals_11);  primals_11 = None
    mul_361: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_360, 256)
    sum_154: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True)
    mul_362: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_360, mul_2);  mul_360 = None
    sum_155: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [2], True);  mul_362 = None
    mul_363: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_2, sum_155);  sum_155 = None
    sub_103: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_361, sum_154);  mul_361 = sum_154 = None
    sub_104: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_103, mul_363);  sub_103 = mul_363 = None
    mul_364: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_104);  div_25 = sub_104 = None
    mul_365: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_336, mul_2);  mul_2 = None
    sum_156: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 1]);  mul_365 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_336, [0, 1]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_148: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_145, mul_364);  add_145 = mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_337: "f32[7696, 256]" = torch.ops.aten.reshape.default(add_148, [7696, 256])
    mm_108: "f32[7696, 256]" = torch.ops.aten.mm.default(view_337, permute_327);  permute_327 = None
    permute_328: "f32[256, 7696]" = torch.ops.aten.permute.default(view_337, [1, 0])
    mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_328, view_5);  permute_328 = view_5 = None
    permute_329: "f32[256, 256]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_158: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
    view_338: "f32[256]" = torch.ops.aten.reshape.default(sum_158, [256]);  sum_158 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_339: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_108, [8, 962, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_340: "f32[8, 962, 4, 64]" = torch.ops.aten.reshape.default(view_339, [8, 962, 4, 64]);  view_339 = None
    permute_331: "f32[8, 4, 962, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_backward_12 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_331, getitem_2, getitem_3, getitem_4, alias_25, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12);  permute_331 = getitem_2 = getitem_3 = getitem_4 = alias_25 = getitem_6 = getitem_7 = getitem_8 = getitem_11 = getitem_12 = None
    getitem_252: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_12[0]
    getitem_253: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_12[1]
    getitem_254: "f32[8, 4, 962, 64]" = _scaled_dot_product_flash_attention_backward_12[2];  _scaled_dot_product_flash_attention_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[24, 4, 962, 64]" = torch.ops.aten.cat.default([getitem_252, getitem_253, getitem_254]);  getitem_252 = getitem_253 = getitem_254 = None
    view_341: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.reshape.default(cat_15, [3, 8, 4, 962, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_332: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.permute.default(view_341, [1, 3, 0, 2, 4]);  view_341 = None
    clone_55: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.clone.default(permute_332, memory_format = torch.contiguous_format);  permute_332 = None
    view_342: "f32[8, 962, 768]" = torch.ops.aten.reshape.default(clone_55, [8, 962, 768]);  clone_55 = None
    view_343: "f32[7696, 768]" = torch.ops.aten.reshape.default(view_342, [7696, 768]);  view_342 = None
    mm_110: "f32[7696, 256]" = torch.ops.aten.mm.default(view_343, permute_333);  permute_333 = None
    permute_334: "f32[768, 7696]" = torch.ops.aten.permute.default(view_343, [1, 0])
    mm_111: "f32[768, 256]" = torch.ops.aten.mm.default(permute_334, view_1);  permute_334 = view_1 = None
    permute_335: "f32[256, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_159: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
    view_344: "f32[768]" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None
    permute_336: "f32[768, 256]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_345: "f32[8, 962, 256]" = torch.ops.aten.reshape.default(mm_110, [8, 962, 256]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    mul_367: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_345, primals_5);  primals_5 = None
    mul_368: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_367, 256)
    sum_160: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_367, mul);  mul_367 = None
    sum_161: "f32[8, 962, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul, sum_161);  sum_161 = None
    sub_106: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(mul_368, sum_160);  mul_368 = sum_160 = None
    sub_107: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(sub_106, mul_370);  sub_106 = mul_370 = None
    div_26: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt, 256);  rsqrt = None
    mul_371: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(div_26, sub_107);  div_26 = sub_107 = None
    mul_372: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(view_345, mul);  mul = None
    sum_162: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_345, [0, 1]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_149: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_148, mul_371);  add_148 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_18: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_149, 1, 0, 1)
    slice_19: "f32[8, 961, 256]" = torch.ops.aten.slice.Tensor(add_149, 1, 1, 962);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    permute_337: "f32[8, 256, 961]" = torch.ops.aten.permute.default(slice_19, [0, 2, 1]);  slice_19 = None
    view_346: "f32[8, 256, 31, 31]" = torch.ops.aten.reshape.default(permute_337, [8, 256, 31, 31]);  permute_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:258, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    sum_164: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(slice_18, [0], True);  slice_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:257, code: x = self.pos_drop(x + self.pos_embed)
    sum_165: "f32[1, 256, 31, 31]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:138, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(view_346, primals_173, primals_3, [256], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_346 = primals_173 = primals_3 = None
    getitem_256: "f32[256, 3, 14, 14]" = convolution_backward_2[1]
    getitem_257: "f32[256]" = convolution_backward_2[2];  convolution_backward_2 = None
    return [sum_165, sum_164, getitem_256, getitem_257, sum_162, sum_163, permute_336, view_344, permute_330, view_338, sum_156, sum_157, permute_326, view_335, permute_322, view_332, sum_150, sum_151, permute_318, view_329, permute_312, view_323, sum_144, sum_145, permute_308, view_320, permute_304, view_317, sum_138, sum_139, permute_300, view_314, permute_294, view_308, sum_132, sum_133, permute_290, view_305, permute_286, view_302, getitem_244, getitem_245, permute_281, view_297, sum_125, sum_126, permute_276, view_294, permute_270, view_288, sum_119, sum_120, permute_266, view_285, permute_262, view_282, sum_113, sum_114, permute_258, view_279, permute_252, view_273, sum_107, sum_108, permute_248, view_270, permute_244, view_267, sum_101, sum_102, permute_240, view_264, permute_234, view_258, sum_95, sum_96, permute_230, view_255, permute_226, view_252, sum_89, sum_90, permute_222, view_249, permute_216, view_243, sum_83, sum_84, permute_212, view_240, permute_208, view_237, sum_77, sum_78, permute_204, view_234, permute_198, view_228, sum_71, sum_72, permute_194, view_225, permute_190, view_222, sum_65, sum_66, permute_186, view_219, permute_180, view_213, sum_59, sum_60, permute_176, view_210, permute_172, view_207, getitem_223, getitem_224, permute_167, view_202, sum_52, sum_53, permute_162, view_199, permute_156, view_193, sum_46, sum_47, permute_152, view_190, permute_148, view_187, sum_40, sum_41, permute_144, view_184, permute_138, view_178, sum_34, sum_35, permute_134, view_175, permute_130, view_172, sum_28, sum_29, permute_126, view_169, permute_120, view_163, sum_22, sum_23, permute_116, view_160, permute_112, view_157, sum_16, sum_17, permute_108, view_154, permute_102, view_148, sum_10, sum_11, permute_98, view_145, permute_94, view_142, sum_4, sum_5, permute_90, view_140, None]
    