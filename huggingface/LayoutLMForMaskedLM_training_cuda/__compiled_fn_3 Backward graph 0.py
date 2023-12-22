from __future__ import annotations



def forward(self, primals_8: "f32[768]", primals_18: "f32[768]", primals_24: "f32[768]", primals_34: "f32[768]", primals_40: "f32[768]", primals_50: "f32[768]", primals_56: "f32[768]", primals_66: "f32[768]", primals_72: "f32[768]", primals_82: "f32[768]", primals_88: "f32[768]", primals_98: "f32[768]", primals_104: "f32[768]", primals_114: "f32[768]", primals_120: "f32[768]", primals_130: "f32[768]", primals_136: "f32[768]", primals_146: "f32[768]", primals_152: "f32[768]", primals_162: "f32[768]", primals_168: "f32[768]", primals_178: "f32[768]", primals_184: "f32[768]", primals_194: "f32[768]", primals_200: "f32[768]", primals_206: "f32[768]", primals_211: "i64[1, 512]", primals_212: "i64[1, 512]", full_default: "i64[1, 512]", slice_1: "i64[1, 512]", select: "i64[1, 512]", select_1: "i64[1, 512]", select_2: "i64[1, 512]", select_3: "i64[1, 512]", mul_1: "f32[1, 512, 768]", getitem_3: "b8[1, 512, 768]", view: "f32[512, 768]", clone_default_33: "f32[1, 12, 512, 64]", clone_default_34: "f32[1, 12, 512, 64]", clone_default_35: "f32[1, 12, 512, 64]", getitem_204: "f32[1, 12, 512]", getitem_205: "i64[]", getitem_206: "i64[]", alias_default_23: "f32[1, 12, 512, 64]", view_16: "f32[512, 768]", getitem_7: "b8[1, 512, 768]", mul_3: "f32[1, 512, 768]", view_18: "f32[512, 768]", addmm_4: "f32[512, 3072]", view_20: "f32[512, 3072]", getitem_11: "b8[1, 512, 768]", mul_8: "f32[1, 512, 768]", view_22: "f32[512, 768]", clone_default_30: "f32[1, 12, 512, 64]", clone_default_31: "f32[1, 12, 512, 64]", clone_default_32: "f32[1, 12, 512, 64]", getitem_197: "f32[1, 12, 512]", getitem_198: "i64[]", getitem_199: "i64[]", alias_default_21: "f32[1, 12, 512, 64]", view_38: "f32[512, 768]", getitem_17: "b8[1, 512, 768]", mul_10: "f32[1, 512, 768]", view_40: "f32[512, 768]", addmm_10: "f32[512, 3072]", view_42: "f32[512, 3072]", getitem_21: "b8[1, 512, 768]", mul_15: "f32[1, 512, 768]", view_44: "f32[512, 768]", clone_default_27: "f32[1, 12, 512, 64]", clone_default_28: "f32[1, 12, 512, 64]", clone_default_29: "f32[1, 12, 512, 64]", getitem_190: "f32[1, 12, 512]", getitem_191: "i64[]", getitem_192: "i64[]", alias_default_19: "f32[1, 12, 512, 64]", view_60: "f32[512, 768]", getitem_27: "b8[1, 512, 768]", mul_17: "f32[1, 512, 768]", view_62: "f32[512, 768]", addmm_16: "f32[512, 3072]", view_64: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_22: "f32[1, 512, 768]", view_66: "f32[512, 768]", clone_default_24: "f32[1, 12, 512, 64]", clone_default_25: "f32[1, 12, 512, 64]", clone_default_26: "f32[1, 12, 512, 64]", getitem_183: "f32[1, 12, 512]", getitem_184: "i64[]", getitem_185: "i64[]", alias_default_17: "f32[1, 12, 512, 64]", view_82: "f32[512, 768]", getitem_37: "b8[1, 512, 768]", mul_24: "f32[1, 512, 768]", view_84: "f32[512, 768]", addmm_22: "f32[512, 3072]", view_86: "f32[512, 3072]", getitem_41: "b8[1, 512, 768]", mul_29: "f32[1, 512, 768]", view_88: "f32[512, 768]", clone_default_21: "f32[1, 12, 512, 64]", clone_default_22: "f32[1, 12, 512, 64]", clone_default_23: "f32[1, 12, 512, 64]", getitem_176: "f32[1, 12, 512]", getitem_177: "i64[]", getitem_178: "i64[]", alias_default_15: "f32[1, 12, 512, 64]", view_104: "f32[512, 768]", getitem_47: "b8[1, 512, 768]", mul_31: "f32[1, 512, 768]", view_106: "f32[512, 768]", addmm_28: "f32[512, 3072]", view_108: "f32[512, 3072]", getitem_51: "b8[1, 512, 768]", mul_36: "f32[1, 512, 768]", view_110: "f32[512, 768]", clone_default_18: "f32[1, 12, 512, 64]", clone_default_19: "f32[1, 12, 512, 64]", clone_default_20: "f32[1, 12, 512, 64]", getitem_169: "f32[1, 12, 512]", getitem_170: "i64[]", getitem_171: "i64[]", alias_default_13: "f32[1, 12, 512, 64]", view_126: "f32[512, 768]", getitem_57: "b8[1, 512, 768]", mul_38: "f32[1, 512, 768]", view_128: "f32[512, 768]", addmm_34: "f32[512, 3072]", view_130: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_43: "f32[1, 512, 768]", view_132: "f32[512, 768]", clone_default_15: "f32[1, 12, 512, 64]", clone_default_16: "f32[1, 12, 512, 64]", clone_default_17: "f32[1, 12, 512, 64]", getitem_162: "f32[1, 12, 512]", getitem_163: "i64[]", getitem_164: "i64[]", alias_default_11: "f32[1, 12, 512, 64]", view_148: "f32[512, 768]", getitem_67: "b8[1, 512, 768]", mul_45: "f32[1, 512, 768]", view_150: "f32[512, 768]", addmm_40: "f32[512, 3072]", view_152: "f32[512, 3072]", getitem_71: "b8[1, 512, 768]", mul_50: "f32[1, 512, 768]", view_154: "f32[512, 768]", clone_default_12: "f32[1, 12, 512, 64]", clone_default_13: "f32[1, 12, 512, 64]", clone_default_14: "f32[1, 12, 512, 64]", getitem_155: "f32[1, 12, 512]", getitem_156: "i64[]", getitem_157: "i64[]", alias_default_9: "f32[1, 12, 512, 64]", view_170: "f32[512, 768]", getitem_77: "b8[1, 512, 768]", mul_52: "f32[1, 512, 768]", view_172: "f32[512, 768]", addmm_46: "f32[512, 3072]", view_174: "f32[512, 3072]", getitem_81: "b8[1, 512, 768]", mul_57: "f32[1, 512, 768]", view_176: "f32[512, 768]", clone_default_9: "f32[1, 12, 512, 64]", clone_default_10: "f32[1, 12, 512, 64]", clone_default_11: "f32[1, 12, 512, 64]", getitem_148: "f32[1, 12, 512]", getitem_149: "i64[]", getitem_150: "i64[]", alias_default_7: "f32[1, 12, 512, 64]", view_192: "f32[512, 768]", getitem_87: "b8[1, 512, 768]", mul_59: "f32[1, 512, 768]", view_194: "f32[512, 768]", addmm_52: "f32[512, 3072]", view_196: "f32[512, 3072]", getitem_91: "b8[1, 512, 768]", mul_64: "f32[1, 512, 768]", view_198: "f32[512, 768]", clone_default_6: "f32[1, 12, 512, 64]", clone_default_7: "f32[1, 12, 512, 64]", clone_default_8: "f32[1, 12, 512, 64]", getitem_141: "f32[1, 12, 512]", getitem_142: "i64[]", getitem_143: "i64[]", alias_default_5: "f32[1, 12, 512, 64]", view_214: "f32[512, 768]", getitem_97: "b8[1, 512, 768]", mul_66: "f32[1, 512, 768]", view_216: "f32[512, 768]", addmm_58: "f32[512, 3072]", view_218: "f32[512, 3072]", getitem_101: "b8[1, 512, 768]", mul_71: "f32[1, 512, 768]", view_220: "f32[512, 768]", clone_default_3: "f32[1, 12, 512, 64]", clone_default_4: "f32[1, 12, 512, 64]", clone_default_5: "f32[1, 12, 512, 64]", getitem_134: "f32[1, 12, 512]", getitem_135: "i64[]", getitem_136: "i64[]", alias_default_3: "f32[1, 12, 512, 64]", view_236: "f32[512, 768]", getitem_107: "b8[1, 512, 768]", mul_73: "f32[1, 512, 768]", view_238: "f32[512, 768]", addmm_64: "f32[512, 3072]", view_240: "f32[512, 3072]", getitem_111: "b8[1, 512, 768]", mul_78: "f32[1, 512, 768]", view_242: "f32[512, 768]", clone_default: "f32[1, 12, 512, 64]", clone_default_1: "f32[1, 12, 512, 64]", clone_default_2: "f32[1, 12, 512, 64]", getitem_127: "f32[1, 12, 512]", getitem_128: "i64[]", getitem_129: "i64[]", alias_default_1: "f32[1, 12, 512, 64]", view_258: "f32[512, 768]", getitem_117: "b8[1, 512, 768]", mul_80: "f32[1, 512, 768]", view_260: "f32[512, 768]", addmm_70: "f32[512, 3072]", view_262: "f32[512, 3072]", getitem_121: "b8[1, 512, 768]", mul_85: "f32[1, 512, 768]", view_264: "f32[512, 768]", addmm_73: "f32[512, 768]", mul_90: "f32[1, 512, 768]", view_266: "f32[512, 768]", sub_42: "f32[512, 30522]", convert_element_type: "f32[]", permute_135: "f32[30522, 768]", div_26: "f32[1, 512, 1]", permute_139: "f32[768, 768]", div_27: "f32[1, 512, 1]", permute_143: "f32[768, 3072]", permute_147: "f32[3072, 768]", div_28: "f32[1, 512, 1]", permute_151: "f32[768, 768]", permute_163: "f32[768, 768]", permute_168: "f32[768, 768]", permute_172: "f32[768, 768]", div_30: "f32[1, 512, 1]", permute_176: "f32[768, 3072]", permute_180: "f32[3072, 768]", div_31: "f32[1, 512, 1]", permute_184: "f32[768, 768]", permute_196: "f32[768, 768]", permute_201: "f32[768, 768]", permute_205: "f32[768, 768]", div_33: "f32[1, 512, 1]", permute_209: "f32[768, 3072]", permute_213: "f32[3072, 768]", div_34: "f32[1, 512, 1]", permute_217: "f32[768, 768]", permute_229: "f32[768, 768]", permute_234: "f32[768, 768]", permute_238: "f32[768, 768]", div_36: "f32[1, 512, 1]", permute_242: "f32[768, 3072]", permute_246: "f32[3072, 768]", div_37: "f32[1, 512, 1]", permute_250: "f32[768, 768]", permute_262: "f32[768, 768]", permute_267: "f32[768, 768]", permute_271: "f32[768, 768]", div_39: "f32[1, 512, 1]", permute_275: "f32[768, 3072]", permute_279: "f32[3072, 768]", div_40: "f32[1, 512, 1]", permute_283: "f32[768, 768]", permute_295: "f32[768, 768]", permute_300: "f32[768, 768]", permute_304: "f32[768, 768]", div_42: "f32[1, 512, 1]", permute_308: "f32[768, 3072]", permute_312: "f32[3072, 768]", div_43: "f32[1, 512, 1]", permute_316: "f32[768, 768]", permute_328: "f32[768, 768]", permute_333: "f32[768, 768]", permute_337: "f32[768, 768]", div_45: "f32[1, 512, 1]", permute_341: "f32[768, 3072]", permute_345: "f32[3072, 768]", div_46: "f32[1, 512, 1]", permute_349: "f32[768, 768]", permute_361: "f32[768, 768]", permute_366: "f32[768, 768]", permute_370: "f32[768, 768]", div_48: "f32[1, 512, 1]", permute_374: "f32[768, 3072]", permute_378: "f32[3072, 768]", div_49: "f32[1, 512, 1]", permute_382: "f32[768, 768]", permute_394: "f32[768, 768]", permute_399: "f32[768, 768]", permute_403: "f32[768, 768]", div_51: "f32[1, 512, 1]", permute_407: "f32[768, 3072]", permute_411: "f32[3072, 768]", div_52: "f32[1, 512, 1]", permute_415: "f32[768, 768]", permute_427: "f32[768, 768]", permute_432: "f32[768, 768]", permute_436: "f32[768, 768]", div_54: "f32[1, 512, 1]", permute_440: "f32[768, 3072]", permute_444: "f32[3072, 768]", div_55: "f32[1, 512, 1]", permute_448: "f32[768, 768]", permute_460: "f32[768, 768]", permute_465: "f32[768, 768]", permute_469: "f32[768, 768]", div_57: "f32[1, 512, 1]", permute_473: "f32[768, 3072]", permute_477: "f32[3072, 768]", div_58: "f32[1, 512, 1]", permute_481: "f32[768, 768]", permute_493: "f32[768, 768]", permute_498: "f32[768, 768]", permute_502: "f32[768, 768]", div_60: "f32[1, 512, 1]", permute_506: "f32[768, 3072]", permute_510: "f32[3072, 768]", div_61: "f32[1, 512, 1]", permute_514: "f32[768, 768]", permute_526: "f32[768, 768]", permute_531: "f32[768, 768]", permute_535: "f32[768, 768]", div_63: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 30522]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_14: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_22: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_30: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_38: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_46: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_54: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_70: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_78: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_86: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_94: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_102: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:576, code: hidden_states = self.dense(hidden_states)
    view_265: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_73, [1, 512, 768]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:970, code: labels.view(-1),
    view_269: "i64[512]" = torch.ops.aten.view.default(primals_212, [-1]);  primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:968, code: masked_lm_loss = loss_fct(
    alias_13: "f32[512, 30522]" = torch.ops.aten.alias.default(sub_42);  sub_42 = None
    full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    unsqueeze_3: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_269, 1);  view_269 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
    where_2: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_4);  unsqueeze_3 = full_default_4 = None
    full_default_7: "f32[512, 30522]" = torch.ops.aten.full.default([512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[512, 30522]" = torch.ops.aten.scatter.value(full_default_7, 1, where_2, -1.0);  full_default_7 = where_2 = None
    where_3: "f32[512, 1]" = torch.ops.aten.where.self(ne_3, div_25, full_default_5);  ne_3 = div_25 = None
    mul_92: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    alias_14: "f32[512, 30522]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    exp_13: "f32[512, 30522]" = torch.ops.aten.exp.default(alias_14);  alias_14 = None
    sum_16: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [1], True)
    mul_93: "f32[512, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_43: "f32[512, 30522]" = torch.ops.aten.sub.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:969, code: prediction_scores.view(-1, self.config.vocab_size),
    view_270: "f32[1, 512, 30522]" = torch.ops.aten.view.default(sub_43, [1, 512, 30522]);  sub_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:969, code: prediction_scores.view(-1, self.config.vocab_size),
    add_109: "f32[1, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, view_270);  tangents_2 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:599, code: hidden_states = self.decoder(hidden_states)
    view_271: "f32[512, 30522]" = torch.ops.aten.view.default(add_109, [512, 30522]);  add_109 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_271, permute_135);  permute_135 = None
    permute_136: "f32[30522, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_1: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_136, view_266);  permute_136 = view_266 = None
    permute_137: "f32[768, 30522]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[30522]" = torch.ops.aten.view.default(sum_17, [30522]);  sum_17 = None
    permute_138: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_273: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:578, code: hidden_states = self.LayerNorm(hidden_states)
    mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_206);  primals_206 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_95, 768)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_95, mul_90);  mul_95 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_19);  sum_19 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_96, sum_18);  mul_96 = sum_18 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_98);  sub_45 = mul_98 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_46);  div_26 = sub_46 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_90);  mul_90 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_102: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
    mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_103, -0.5);  mul_103 = None
    exp_14: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_104);  mul_104 = None
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_105);  view_265 = mul_105 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_102, mul_106);  mul_102 = mul_106 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_99, add_111);  mul_99 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:576, code: hidden_states = self.dense(hidden_states)
    view_274: "f32[512, 768]" = torch.ops.aten.view.default(mul_107, [512, 768]);  mul_107 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_274, permute_139);  permute_139 = None
    permute_140: "f32[768, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_140, view_264);  permute_140 = view_264 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    view_276: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_2, [1, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, primals_200);  primals_200 = None
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_109, 768)
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_109, mul_85);  mul_109 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_24);  sum_24 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_110, sum_23);  mul_110 = sum_23 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_112);  sub_48 = mul_112 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_49);  div_27 = sub_49 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_85);  mul_85 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_113, mul_115);  mul_115 = None
    clone_12: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_116, memory_format = torch.contiguous_format);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[512, 768]" = torch.ops.aten.view.default(clone_12, [512, 768]);  clone_12 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_277, permute_143);  permute_143 = None
    permute_144: "f32[768, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_144, view_262);  permute_144 = view_262 = None
    permute_145: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_146: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_279: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_119: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_120: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_119, -0.5);  mul_119 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_120);  mul_120 = None
    mul_121: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_122: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_121);  view_261 = mul_121 = None
    add_113: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_118, mul_122);  mul_118 = mul_122 = None
    mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_113);  view_279 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 3072]" = torch.ops.aten.view.default(mul_123, [512, 3072]);  mul_123 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_280, permute_147);  permute_147 = None
    permute_148: "f32[3072, 512]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_148, view_260);  permute_148 = view_260 = None
    permute_149: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[3072]" = torch.ops.aten.view.default(sum_28, [3072]);  sum_28 = None
    permute_150: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_282: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_113, view_282);  mul_113 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_194);  primals_194 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, 768)
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True)
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, mul_80);  mul_125 = None
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True);  mul_127 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_30);  sum_30 = None
    sub_51: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_126, sum_29);  mul_126 = sum_29 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_51, mul_128);  sub_51 = mul_128 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_52);  div_28 = sub_52 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_80);  mul_80 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1]);  mul_130 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_131);  mul_131 = None
    clone_13: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_132, memory_format = torch.contiguous_format);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[512, 768]" = torch.ops.aten.view.default(clone_13, [512, 768]);  clone_13 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_283, permute_151);  permute_151 = None
    permute_152: "f32[768, 512]" = torch.ops.aten.permute.default(view_283, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_258);  permute_152 = view_258 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
    view_284: "f32[768]" = torch.ops.aten.view.default(sum_33, [768]);  sum_33 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_285: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_286: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_285, [1, 512, 12, 64]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_155: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_155, clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_155 = clone_default = clone_default_1 = clone_default_2 = alias_default_1 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_130: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
    getitem_131: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
    getitem_132: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_15, [1, 512, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_294: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_16, [1, 512, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_295: "f32[512, 768]" = torch.ops.aten.view.default(view_294, [512, 768]);  view_294 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_295, permute_163);  permute_163 = None
    permute_164: "f32[768, 512]" = torch.ops.aten.permute.default(view_295, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_164, view_242);  permute_164 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
    view_296: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_297: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_115: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_129, view_297);  mul_129 = view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_298: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_167, [1, 512, 768]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_299: "f32[512, 768]" = torch.ops.aten.view.default(view_298, [512, 768]);  view_298 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_299, permute_168);  permute_168 = None
    permute_169: "f32[768, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_169, view_242);  permute_169 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_171: "f32[768, 768]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_301: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_115, view_301);  add_115 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_302: "f32[512, 768]" = torch.ops.aten.view.default(view_293, [512, 768]);  view_293 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_302, permute_172);  permute_172 = None
    permute_173: "f32[768, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_173, view_242);  permute_173 = view_242 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[768]" = torch.ops.aten.view.default(sum_37, [768]);  sum_37 = None
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_304: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_116, view_304);  add_116 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_184);  primals_184 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, 768)
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_138, mul_78);  mul_138 = None
    sum_39: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_39);  sum_39 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_139, sum_38);  mul_139 = sum_38 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_141);  sub_55 = mul_141 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_56);  div_30 = sub_56 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_78);  mul_78 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_142, mul_144);  mul_144 = None
    clone_17: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_145, memory_format = torch.contiguous_format);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[512, 768]" = torch.ops.aten.view.default(clone_17, [512, 768]);  clone_17 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_305, permute_176);  permute_176 = None
    permute_177: "f32[768, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_240);  permute_177 = view_240 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_307: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_147: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
    mul_148: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_149: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_148, -0.5);  mul_148 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_149);  mul_149 = None
    mul_150: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_151: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_150);  view_239 = mul_150 = None
    add_119: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_147, mul_151);  mul_147 = mul_151 = None
    mul_152: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_307, add_119);  view_307 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[512, 3072]" = torch.ops.aten.view.default(mul_152, [512, 3072]);  mul_152 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_308, permute_180);  permute_180 = None
    permute_181: "f32[3072, 512]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_238);  permute_181 = view_238 = None
    permute_182: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[3072]" = torch.ops.aten.view.default(sum_43, [3072]);  sum_43 = None
    permute_183: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_310: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_120: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_142, view_310);  mul_142 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_178);  primals_178 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, 768)
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_73);  mul_154 = None
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_45);  sum_45 = None
    sub_58: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_155, sum_44);  mul_155 = sum_44 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_58, mul_157);  sub_58 = mul_157 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_59);  div_31 = sub_59 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_73);  mul_73 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_158, mul_160);  mul_160 = None
    clone_18: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_161, memory_format = torch.contiguous_format);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_311: "f32[512, 768]" = torch.ops.aten.view.default(clone_18, [512, 768]);  clone_18 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_311, permute_184);  permute_184 = None
    permute_185: "f32[768, 512]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_236);  permute_185 = view_236 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_313: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_314: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_313, [1, 512, 12, 64]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_188: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_188, clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale = 0.125);  permute_188 = clone_default_3 = clone_default_4 = clone_default_5 = alias_default_3 = getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
    getitem_138: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
    getitem_139: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_20, [1, 512, 768]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    view_322: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_21, [1, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_323: "f32[512, 768]" = torch.ops.aten.view.default(view_322, [512, 768]);  view_322 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_323, permute_196);  permute_196 = None
    permute_197: "f32[768, 512]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_197, view_220);  permute_197 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    view_325: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_121: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_158, view_325);  mul_158 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_326: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_200, [1, 512, 768]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_327: "f32[512, 768]" = torch.ops.aten.view.default(view_326, [512, 768]);  view_326 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_327, permute_201);  permute_201 = None
    permute_202: "f32[768, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_202, view_220);  permute_202 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    view_329: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_329);  add_121 = view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_330: "f32[512, 768]" = torch.ops.aten.view.default(view_321, [512, 768]);  view_321 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_330, permute_205);  permute_205 = None
    permute_206: "f32[768, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_206, view_220);  permute_206 = view_220 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[768]" = torch.ops.aten.view.default(sum_52, [768]);  sum_52 = None
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_332: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_122, view_332);  add_122 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_168);  primals_168 = None
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_71);  mul_167 = None
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, sum_54);  sum_54 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_53);  mul_168 = sum_53 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_170);  sub_62 = mul_170 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_63);  div_33 = sub_63 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_71);  mul_71 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_173);  mul_173 = None
    clone_22: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_174, memory_format = torch.contiguous_format);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[512, 768]" = torch.ops.aten.view.default(clone_22, [512, 768]);  clone_22 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_333, permute_209);  permute_209 = None
    permute_210: "f32[768, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_218);  permute_210 = view_218 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_335: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_176: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_177: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_178: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_177, -0.5);  mul_177 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_178);  mul_178 = None
    mul_179: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_180: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_179);  view_217 = mul_179 = None
    add_125: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_176, mul_180);  mul_176 = mul_180 = None
    mul_181: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_335, add_125);  view_335 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_336: "f32[512, 3072]" = torch.ops.aten.view.default(mul_181, [512, 3072]);  mul_181 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_336, permute_213);  permute_213 = None
    permute_214: "f32[3072, 512]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_216);  permute_214 = view_216 = None
    permute_215: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
    view_337: "f32[3072]" = torch.ops.aten.view.default(sum_58, [3072]);  sum_58 = None
    permute_216: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_338: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_171, view_338);  mul_171 = view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_162);  primals_162 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, 768)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True)
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, mul_66);  mul_183 = None
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_60);  sum_60 = None
    sub_65: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_184, sum_59);  mul_184 = sum_59 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_186);  sub_65 = mul_186 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_66);  div_34 = sub_66 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_66);  mul_66 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1]);  mul_188 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_187, mul_189);  mul_189 = None
    clone_23: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_190, memory_format = torch.contiguous_format);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 768]" = torch.ops.aten.view.default(clone_23, [512, 768]);  clone_23 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_339, permute_217);  permute_217 = None
    permute_218: "f32[768, 512]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_214);  permute_218 = view_214 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_342: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_341, [1, 512, 12, 64]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_221: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_221, clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale = 0.125);  permute_221 = clone_default_6 = clone_default_7 = clone_default_8 = alias_default_5 = getitem_141 = getitem_142 = getitem_143 = None
    getitem_144: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
    getitem_145: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
    getitem_146: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_25, [1, 512, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_350: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_26, [1, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_351: "f32[512, 768]" = torch.ops.aten.view.default(view_350, [512, 768]);  view_350 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_351, permute_229);  permute_229 = None
    permute_230: "f32[768, 512]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_230, view_198);  permute_230 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.view.default(sum_65, [768]);  sum_65 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_127: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_187, view_353);  mul_187 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_233: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_354: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_233, [1, 512, 768]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_355: "f32[512, 768]" = torch.ops.aten.view.default(view_354, [512, 768]);  view_354 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_355, permute_234);  permute_234 = None
    permute_235: "f32[768, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_235, view_198);  permute_235 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_357: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_128: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_127, view_357);  add_127 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_358: "f32[512, 768]" = torch.ops.aten.view.default(view_349, [512, 768]);  view_349 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_358, permute_238);  permute_238 = None
    permute_239: "f32[768, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_198);  permute_239 = view_198 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_360: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_128, view_360);  add_128 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_152);  primals_152 = None
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, 768)
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True)
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_196, mul_64);  mul_196 = None
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True);  mul_198 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_69);  sum_69 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_197, sum_68);  mul_197 = sum_68 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_199);  sub_69 = mul_199 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_70);  div_36 = sub_70 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_64);  mul_64 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_201, [0, 1]);  mul_201 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_200, mul_202);  mul_202 = None
    clone_27: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_203, memory_format = torch.contiguous_format);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[512, 768]" = torch.ops.aten.view.default(clone_27, [512, 768]);  clone_27 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_361, permute_242);  permute_242 = None
    permute_243: "f32[768, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_243, view_196);  permute_243 = view_196 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_245: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_363: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_205: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_207: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_206, -0.5);  mul_206 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_207);  mul_207 = None
    mul_208: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_209: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_208);  view_195 = mul_208 = None
    add_131: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_205, mul_209);  mul_205 = mul_209 = None
    mul_210: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_363, add_131);  view_363 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 3072]" = torch.ops.aten.view.default(mul_210, [512, 3072]);  mul_210 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_364, permute_246);  permute_246 = None
    permute_247: "f32[3072, 512]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_247, view_194);  permute_247 = view_194 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[3072]" = torch.ops.aten.view.default(sum_73, [3072]);  sum_73 = None
    permute_249: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_200, view_366);  mul_200 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_146);  primals_146 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, 768)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True)
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, mul_59);  mul_212 = None
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_75);  sum_75 = None
    sub_72: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_213, sum_74);  mul_213 = sum_74 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_72, mul_215);  sub_72 = mul_215 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_73);  div_37 = sub_73 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_59);  mul_59 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1]);  mul_217 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_218);  mul_218 = None
    clone_28: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_219, memory_format = torch.contiguous_format);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_367: "f32[512, 768]" = torch.ops.aten.view.default(clone_28, [512, 768]);  clone_28 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_367, permute_250);  permute_250 = None
    permute_251: "f32[768, 512]" = torch.ops.aten.permute.default(view_367, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_192);  permute_251 = view_192 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
    view_368: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_369: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_370: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_369, [1, 512, 12, 64]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_254: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_254, clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale = 0.125);  permute_254 = clone_default_9 = clone_default_10 = clone_default_11 = alias_default_7 = getitem_148 = getitem_149 = getitem_150 = None
    getitem_151: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
    getitem_152: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
    getitem_153: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_30, [1, 512, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_31, [1, 512, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_379: "f32[512, 768]" = torch.ops.aten.view.default(view_378, [512, 768]);  view_378 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_379, permute_262);  permute_262 = None
    permute_263: "f32[768, 512]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_263, view_176);  permute_263 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_381: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_216, view_381);  mul_216 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_382: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_266, [1, 512, 768]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_383: "f32[512, 768]" = torch.ops.aten.view.default(view_382, [512, 768]);  view_382 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_383, permute_267);  permute_267 = None
    permute_268: "f32[768, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_268, view_176);  permute_268 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_270: "f32[768, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_385: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_133, view_385);  add_133 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_386: "f32[512, 768]" = torch.ops.aten.view.default(view_377, [512, 768]);  view_377 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_386, permute_271);  permute_271 = None
    permute_272: "f32[768, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_176);  permute_272 = view_176 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    view_388: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_134, view_388);  add_134 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, primals_136);  primals_136 = None
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, 768)
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_225, mul_57);  mul_225 = None
    sum_84: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_84);  sum_84 = None
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_226, sum_83);  mul_226 = sum_83 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_228);  sub_76 = mul_228 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_77);  div_39 = sub_77 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_57);  mul_57 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_229, mul_231);  mul_231 = None
    clone_32: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_232, memory_format = torch.contiguous_format);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 768]" = torch.ops.aten.view.default(clone_32, [512, 768]);  clone_32 = None
    mm_52: "f32[512, 3072]" = torch.ops.aten.mm.default(view_389, permute_275);  permute_275 = None
    permute_276: "f32[768, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_276, view_174);  permute_276 = view_174 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_391: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_52, [1, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_234: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
    mul_235: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_237);  view_173 = mul_237 = None
    add_137: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_391, add_137);  view_391 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 3072]" = torch.ops.aten.view.default(mul_239, [512, 3072]);  mul_239 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_392, permute_279);  permute_279 = None
    permute_280: "f32[3072, 512]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_280, view_172);  permute_280 = view_172 = None
    permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[3072]" = torch.ops.aten.view.default(sum_88, [3072]);  sum_88 = None
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_394: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_138: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_229, view_394);  mul_229 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_130);  primals_130 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, 768)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_52);  mul_241 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_90);  sum_90 = None
    sub_79: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_242, sum_89);  mul_242 = sum_89 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_79, mul_244);  sub_79 = mul_244 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_80);  div_40 = sub_80 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_52);  mul_52 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_245, mul_247);  mul_247 = None
    clone_33: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_248, memory_format = torch.contiguous_format);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_395: "f32[512, 768]" = torch.ops.aten.view.default(clone_33, [512, 768]);  clone_33 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_395, permute_283);  permute_283 = None
    permute_284: "f32[768, 512]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_170);  permute_284 = view_170 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[768]" = torch.ops.aten.view.default(sum_93, [768]);  sum_93 = None
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_397: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_398: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_397, [1, 512, 12, 64]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_287: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_287, clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale = 0.125);  permute_287 = clone_default_12 = clone_default_13 = clone_default_14 = alias_default_9 = getitem_155 = getitem_156 = getitem_157 = None
    getitem_158: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
    getitem_159: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
    getitem_160: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_35, [1, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_406: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_36, [1, 512, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_407: "f32[512, 768]" = torch.ops.aten.view.default(view_406, [512, 768]);  view_406 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_407, permute_295);  permute_295 = None
    permute_296: "f32[768, 512]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_296, view_154);  permute_296 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    permute_298: "f32[768, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_409: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_139: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_245, view_409);  mul_245 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_299: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_410: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_299, [1, 512, 768]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_411: "f32[512, 768]" = torch.ops.aten.view.default(view_410, [512, 768]);  view_410 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_411, permute_300);  permute_300 = None
    permute_301: "f32[768, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_301, view_154);  permute_301 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_303: "f32[768, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_413: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_413);  add_139 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_414: "f32[512, 768]" = torch.ops.aten.view.default(view_405, [512, 768]);  view_405 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_414, permute_304);  permute_304 = None
    permute_305: "f32[768, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_305, view_154);  permute_305 = view_154 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.view.default(sum_97, [768]);  sum_97 = None
    permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_416: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_140, view_416);  add_140 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_120);  primals_120 = None
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_254, 768)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_254, mul_50);  mul_254 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_99);  sum_99 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_255, sum_98);  mul_255 = sum_98 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_257);  sub_83 = mul_257 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_84);  div_42 = sub_84 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_50);  mul_50 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_261: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_258, mul_260);  mul_260 = None
    clone_37: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_261, memory_format = torch.contiguous_format);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[512, 768]" = torch.ops.aten.view.default(clone_37, [512, 768]);  clone_37 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_417, permute_308);  permute_308 = None
    permute_309: "f32[768, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_309, view_152);  permute_309 = view_152 = None
    permute_310: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_419: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_264, -0.5);  mul_264 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_265);  mul_265 = None
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_266);  view_151 = mul_266 = None
    add_143: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_263, mul_267);  mul_263 = mul_267 = None
    mul_268: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_419, add_143);  view_419 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_420: "f32[512, 3072]" = torch.ops.aten.view.default(mul_268, [512, 3072]);  mul_268 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_420, permute_312);  permute_312 = None
    permute_313: "f32[3072, 512]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_313, view_150);  permute_313 = view_150 = None
    permute_314: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[3072]" = torch.ops.aten.view.default(sum_103, [3072]);  sum_103 = None
    permute_315: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_422: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_144: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_258, view_422);  mul_258 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_114);  primals_114 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, 768)
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, mul_45);  mul_270 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_105);  sum_105 = None
    sub_86: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_271, sum_104);  mul_271 = sum_104 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_273);  sub_86 = mul_273 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_87);  div_43 = sub_87 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_45);  mul_45 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_274, mul_276);  mul_276 = None
    clone_38: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_277, memory_format = torch.contiguous_format);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_423: "f32[512, 768]" = torch.ops.aten.view.default(clone_38, [512, 768]);  clone_38 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_423, permute_316);  permute_316 = None
    permute_317: "f32[768, 512]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_317, view_148);  permute_317 = view_148 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_319: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    view_425: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_426: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_425, [1, 512, 12, 64]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_320: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_320, clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale = 0.125);  permute_320 = clone_default_15 = clone_default_16 = clone_default_17 = alias_default_11 = getitem_162 = getitem_163 = getitem_164 = None
    getitem_165: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
    getitem_166: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
    getitem_167: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_40, [1, 512, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_434: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_41, [1, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_435: "f32[512, 768]" = torch.ops.aten.view.default(view_434, [512, 768]);  view_434 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_435, permute_328);  permute_328 = None
    permute_329: "f32[768, 512]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_329, view_132);  permute_329 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    permute_331: "f32[768, 768]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_437: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_274, view_437);  mul_274 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_332: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_438: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_332, [1, 512, 768]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_439: "f32[512, 768]" = torch.ops.aten.view.default(view_438, [512, 768]);  view_438 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_333);  permute_333 = None
    permute_334: "f32[768, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_334, view_132);  permute_334 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    permute_336: "f32[768, 768]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_441);  add_145 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_442: "f32[512, 768]" = torch.ops.aten.view.default(view_433, [512, 768]);  view_433 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_442, permute_337);  permute_337 = None
    permute_338: "f32[768, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_338, view_132);  permute_338 = view_132 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_444: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_146, view_444);  add_146 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, primals_104);  primals_104 = None
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_283, 768)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_283, mul_43);  mul_283 = None
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, sum_114);  sum_114 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_284, sum_113);  mul_284 = sum_113 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_286);  sub_90 = mul_286 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_91);  div_45 = sub_91 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_43);  mul_43 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_287, mul_289);  mul_289 = None
    clone_42: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_290, memory_format = torch.contiguous_format);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 768]" = torch.ops.aten.view.default(clone_42, [512, 768]);  clone_42 = None
    mm_76: "f32[512, 3072]" = torch.ops.aten.mm.default(view_445, permute_341);  permute_341 = None
    permute_342: "f32[768, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_342, view_130);  permute_342 = view_130 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
    permute_344: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    view_447: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_76, [1, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_292: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
    mul_293: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_294: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_293, -0.5);  mul_293 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_294);  mul_294 = None
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_295);  view_129 = mul_295 = None
    add_149: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_292, mul_296);  mul_292 = mul_296 = None
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_149);  view_447 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 3072]" = torch.ops.aten.view.default(mul_297, [512, 3072]);  mul_297 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_448, permute_345);  permute_345 = None
    permute_346: "f32[3072, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_346, view_128);  permute_346 = view_128 = None
    permute_347: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[3072]" = torch.ops.aten.view.default(sum_118, [3072]);  sum_118 = None
    permute_348: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_287, view_450);  mul_287 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_98);  primals_98 = None
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, 768)
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, mul_38);  mul_299 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_120);  sum_120 = None
    sub_93: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_300, sum_119);  mul_300 = sum_119 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_93, mul_302);  sub_93 = mul_302 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_94);  div_46 = sub_94 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_38);  mul_38 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_303, mul_305);  mul_305 = None
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 768]" = torch.ops.aten.view.default(clone_43, [512, 768]);  clone_43 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_349);  permute_349 = None
    permute_350: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_350, view_126);  permute_350 = view_126 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_123, [768]);  sum_123 = None
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_454: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_453, [1, 512, 12, 64]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_353: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_353, clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale = 0.125);  permute_353 = clone_default_18 = clone_default_19 = clone_default_20 = alias_default_13 = getitem_169 = getitem_170 = getitem_171 = None
    getitem_172: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
    getitem_173: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
    getitem_174: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_45, [1, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_462: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_46, [1, 512, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_463: "f32[512, 768]" = torch.ops.aten.view.default(view_462, [512, 768]);  view_462 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_463, permute_361);  permute_361 = None
    permute_362: "f32[768, 512]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_362, view_110);  permute_362 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[768]" = torch.ops.aten.view.default(sum_125, [768]);  sum_125 = None
    permute_364: "f32[768, 768]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_465: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_151: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_303, view_465);  mul_303 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_365: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_365, [1, 512, 768]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_467: "f32[512, 768]" = torch.ops.aten.view.default(view_466, [512, 768]);  view_466 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_467, permute_366);  permute_366 = None
    permute_367: "f32[768, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_367, view_110);  permute_367 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_369: "f32[768, 768]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_469: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_151, view_469);  add_151 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_470: "f32[512, 768]" = torch.ops.aten.view.default(view_461, [512, 768]);  view_461 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_470, permute_370);  permute_370 = None
    permute_371: "f32[768, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_371, view_110);  permute_371 = view_110 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_472: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_152, view_472);  add_152 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_88);  primals_88 = None
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, 768)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_312, mul_36);  mul_312 = None
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_129);  sum_129 = None
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_313, sum_128);  mul_313 = sum_128 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_315);  sub_97 = mul_315 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_98);  div_48 = sub_98 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_36);  mul_36 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None
    clone_47: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_319, memory_format = torch.contiguous_format);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 768]" = torch.ops.aten.view.default(clone_47, [512, 768]);  clone_47 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_473, permute_374);  permute_374 = None
    permute_375: "f32[768, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_375, view_108);  permute_375 = view_108 = None
    permute_376: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_377: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_475: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_321: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_322: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_323: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_322, -0.5);  mul_322 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_323);  mul_323 = None
    mul_324: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_325: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_324);  view_107 = mul_324 = None
    add_155: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_321, mul_325);  mul_321 = mul_325 = None
    mul_326: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_155);  view_475 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[512, 3072]" = torch.ops.aten.view.default(mul_326, [512, 3072]);  mul_326 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_476, permute_378);  permute_378 = None
    permute_379: "f32[3072, 512]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_379, view_106);  permute_379 = view_106 = None
    permute_380: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[3072]" = torch.ops.aten.view.default(sum_133, [3072]);  sum_133 = None
    permute_381: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_316, view_478);  mul_316 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_82);  primals_82 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, 768)
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True)
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, mul_31);  mul_328 = None
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, sum_135);  sum_135 = None
    sub_100: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_329, sum_134);  mul_329 = sum_134 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_331);  sub_100 = mul_331 = None
    mul_332: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_101);  div_49 = sub_101 = None
    mul_333: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_31);  mul_31 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1]);  mul_333 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_334: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_335: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_334);  mul_334 = None
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_335, memory_format = torch.contiguous_format);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_479: "f32[512, 768]" = torch.ops.aten.view.default(clone_48, [512, 768]);  clone_48 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_479, permute_382);  permute_382 = None
    permute_383: "f32[768, 512]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_383, view_104);  permute_383 = view_104 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    view_481: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_482: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_481, [1, 512, 12, 64]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_386: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_386, clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale = 0.125);  permute_386 = clone_default_21 = clone_default_22 = clone_default_23 = alias_default_15 = getitem_176 = getitem_177 = getitem_178 = None
    getitem_179: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
    getitem_180: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
    getitem_181: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_50, [1, 512, 768]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_490: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_51, [1, 512, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_491: "f32[512, 768]" = torch.ops.aten.view.default(view_490, [512, 768]);  view_490 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_491, permute_394);  permute_394 = None
    permute_395: "f32[768, 512]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_395, view_88);  permute_395 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_397: "f32[768, 768]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_493: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_157: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_332, view_493);  mul_332 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_398: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_494: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_398, [1, 512, 768]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_495: "f32[512, 768]" = torch.ops.aten.view.default(view_494, [512, 768]);  view_494 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_495, permute_399);  permute_399 = None
    permute_400: "f32[768, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_400, view_88);  permute_400 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
    permute_402: "f32[768, 768]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    view_497: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_158: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_157, view_497);  add_157 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_498: "f32[512, 768]" = torch.ops.aten.view.default(view_489, [512, 768]);  view_489 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_498, permute_403);  permute_403 = None
    permute_404: "f32[768, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_404, view_88);  permute_404 = view_88 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    permute_406: "f32[768, 768]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    view_500: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_158, view_500);  add_158 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_72);  primals_72 = None
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, 768)
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_341, mul_29);  mul_341 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_144);  sum_144 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_342, sum_143);  mul_342 = sum_143 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_344);  sub_104 = mul_344 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_105);  div_51 = sub_105 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_29);  mul_29 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_345, mul_347);  mul_347 = None
    clone_52: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_348, memory_format = torch.contiguous_format);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 768]" = torch.ops.aten.view.default(clone_52, [512, 768]);  clone_52 = None
    mm_100: "f32[512, 3072]" = torch.ops.aten.mm.default(view_501, permute_407);  permute_407 = None
    permute_408: "f32[768, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_408, view_86);  permute_408 = view_86 = None
    permute_409: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_410: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    view_503: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_100, [1, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_350: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
    mul_351: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_352: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_351, -0.5);  mul_351 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_352);  mul_352 = None
    mul_353: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_354: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_353);  view_85 = mul_353 = None
    add_161: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_350, mul_354);  mul_350 = mul_354 = None
    mul_355: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, add_161);  view_503 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 3072]" = torch.ops.aten.view.default(mul_355, [512, 3072]);  mul_355 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_504, permute_411);  permute_411 = None
    permute_412: "f32[3072, 512]" = torch.ops.aten.permute.default(view_504, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_412, view_84);  permute_412 = view_84 = None
    permute_413: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
    view_505: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    permute_414: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    view_506: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_345, view_506);  mul_345 = view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, primals_66);  primals_66 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, 768)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_24);  mul_357 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_150);  sum_150 = None
    sub_107: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_358, sum_149);  mul_358 = sum_149 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_360);  sub_107 = mul_360 = None
    mul_361: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_108);  div_52 = sub_108 = None
    mul_362: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_24);  mul_24 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_363: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_364: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_361, mul_363);  mul_363 = None
    clone_53: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_364, memory_format = torch.contiguous_format);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_507: "f32[512, 768]" = torch.ops.aten.view.default(clone_53, [512, 768]);  clone_53 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_507, permute_415);  permute_415 = None
    permute_416: "f32[768, 512]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_82);  permute_416 = view_82 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_509: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_510: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_509, [1, 512, 12, 64]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_419: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_419, clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale = 0.125);  permute_419 = clone_default_24 = clone_default_25 = clone_default_26 = alias_default_17 = getitem_183 = getitem_184 = getitem_185 = None
    getitem_186: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
    getitem_187: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
    getitem_188: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_55, [1, 512, 768]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_518: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_56, [1, 512, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_519: "f32[512, 768]" = torch.ops.aten.view.default(view_518, [512, 768]);  view_518 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_519, permute_427);  permute_427 = None
    permute_428: "f32[768, 512]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_428, view_66);  permute_428 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.view.default(sum_155, [768]);  sum_155 = None
    permute_430: "f32[768, 768]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_521: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_361, view_521);  mul_361 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_431: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_522: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_431, [1, 512, 768]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_523: "f32[512, 768]" = torch.ops.aten.view.default(view_522, [512, 768]);  view_522 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_523, permute_432);  permute_432 = None
    permute_433: "f32[768, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_433, view_66);  permute_433 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_435: "f32[768, 768]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_525: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_525);  add_163 = view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_526: "f32[512, 768]" = torch.ops.aten.view.default(view_517, [512, 768]);  view_517 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_526, permute_436);  permute_436 = None
    permute_437: "f32[768, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_437, view_66);  permute_437 = view_66 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[768]" = torch.ops.aten.view.default(sum_157, [768]);  sum_157 = None
    permute_439: "f32[768, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_164, view_528);  add_164 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, primals_56);  primals_56 = None
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, 768)
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_370, mul_22);  mul_370 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_159);  sum_159 = None
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_371, sum_158);  mul_371 = sum_158 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_373);  sub_111 = mul_373 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_112);  div_54 = sub_112 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, mul_22);  mul_22 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, mul_376);  mul_376 = None
    clone_57: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_377, memory_format = torch.contiguous_format);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[512, 768]" = torch.ops.aten.view.default(clone_57, [512, 768]);  clone_57 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_529, permute_440);  permute_440 = None
    permute_441: "f32[768, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_441, view_64);  permute_441 = view_64 = None
    permute_442: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_443: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_531: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_379: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_380: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_380, -0.5);  mul_380 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_381);  mul_381 = None
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_382);  view_63 = mul_382 = None
    add_167: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_379, mul_383);  mul_379 = mul_383 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_531, add_167);  view_531 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_532: "f32[512, 3072]" = torch.ops.aten.view.default(mul_384, [512, 3072]);  mul_384 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_532, permute_444);  permute_444 = None
    permute_445: "f32[3072, 512]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_445, view_62);  permute_445 = view_62 = None
    permute_446: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[3072]" = torch.ops.aten.view.default(sum_163, [3072]);  sum_163 = None
    permute_447: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_168: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_374, view_534);  mul_374 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, primals_50);  primals_50 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, 768)
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_17);  mul_386 = None
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_165);  sum_165 = None
    sub_114: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_387, sum_164);  mul_387 = sum_164 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_389);  sub_114 = mul_389 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_115);  div_55 = sub_115 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, mul_17);  mul_17 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None
    clone_58: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_393, memory_format = torch.contiguous_format);  mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_535: "f32[512, 768]" = torch.ops.aten.view.default(clone_58, [512, 768]);  clone_58 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_535, permute_448);  permute_448 = None
    permute_449: "f32[768, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_60);  permute_449 = view_60 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_537: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_538: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_537, [1, 512, 12, 64]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_452: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_452, clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale = 0.125);  permute_452 = clone_default_27 = clone_default_28 = clone_default_29 = alias_default_19 = getitem_190 = getitem_191 = getitem_192 = None
    getitem_193: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
    getitem_194: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
    getitem_195: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_60, [1, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_546: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_61, [1, 512, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_547: "f32[512, 768]" = torch.ops.aten.view.default(view_546, [512, 768]);  view_546 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_547, permute_460);  permute_460 = None
    permute_461: "f32[768, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_461, view_44);  permute_461 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_463: "f32[768, 768]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_549: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_169: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_390, view_549);  mul_390 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_464: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_550: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_464, [1, 512, 768]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_551: "f32[512, 768]" = torch.ops.aten.view.default(view_550, [512, 768]);  view_550 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_551, permute_465);  permute_465 = None
    permute_466: "f32[768, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_466, view_44);  permute_466 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_468: "f32[768, 768]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    view_553: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_553);  add_169 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_554: "f32[512, 768]" = torch.ops.aten.view.default(view_545, [512, 768]);  view_545 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_554, permute_469);  permute_469 = None
    permute_470: "f32[768, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_470, view_44);  permute_470 = view_44 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    view_556: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_171: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_170, view_556);  add_170 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, primals_40);  primals_40 = None
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, 768)
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_399, mul_15);  mul_399 = None
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_174);  sum_174 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_400, sum_173);  mul_400 = sum_173 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_402);  sub_118 = mul_402 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_119);  div_57 = sub_119 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_15);  mul_15 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_403, mul_405);  mul_405 = None
    clone_62: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_406, memory_format = torch.contiguous_format);  mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[512, 768]" = torch.ops.aten.view.default(clone_62, [512, 768]);  clone_62 = None
    mm_124: "f32[512, 3072]" = torch.ops.aten.mm.default(view_557, permute_473);  permute_473 = None
    permute_474: "f32[768, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_474, view_42);  permute_474 = view_42 = None
    permute_475: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[768]" = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
    permute_476: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    view_559: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_124, [1, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_409, -0.5);  mul_409 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_410);  mul_410 = None
    mul_411: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_412: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_411);  view_41 = mul_411 = None
    add_173: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_408, mul_412);  mul_408 = mul_412 = None
    mul_413: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_559, add_173);  view_559 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_560: "f32[512, 3072]" = torch.ops.aten.view.default(mul_413, [512, 3072]);  mul_413 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_477);  permute_477 = None
    permute_478: "f32[3072, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_478, view_40);  permute_478 = view_40 = None
    permute_479: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[3072]" = torch.ops.aten.view.default(sum_178, [3072]);  sum_178 = None
    permute_480: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_403, view_562);  mul_403 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_174, primals_34);  primals_34 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, 768)
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True)
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, mul_10);  mul_415 = None
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True);  mul_417 = None
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_180);  sum_180 = None
    sub_121: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_416, sum_179);  mul_416 = sum_179 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_121, mul_418);  sub_121 = mul_418 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_122);  div_58 = sub_122 = None
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_174, mul_10);  mul_10 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1]);  mul_420 = None
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_174, [0, 1]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_419, mul_421);  mul_421 = None
    clone_63: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_422, memory_format = torch.contiguous_format);  mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_563: "f32[512, 768]" = torch.ops.aten.view.default(clone_63, [512, 768]);  clone_63 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_563, permute_481);  permute_481 = None
    permute_482: "f32[768, 512]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_482, view_38);  permute_482 = view_38 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    permute_484: "f32[768, 768]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_565: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_566: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_565, [1, 512, 12, 64]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_485: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_485, clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale = 0.125);  permute_485 = clone_default_30 = clone_default_31 = clone_default_32 = alias_default_21 = getitem_197 = getitem_198 = getitem_199 = None
    getitem_200: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
    getitem_201: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
    getitem_202: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_65, [1, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_492, memory_format = torch.contiguous_format);  permute_492 = None
    view_574: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_66, [1, 512, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_575: "f32[512, 768]" = torch.ops.aten.view.default(view_574, [512, 768]);  view_574 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_575, permute_493);  permute_493 = None
    permute_494: "f32[768, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_494, view_22);  permute_494 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
    permute_496: "f32[768, 768]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_577: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_419, view_577);  mul_419 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_497: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_578: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_497, [1, 512, 768]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_579: "f32[512, 768]" = torch.ops.aten.view.default(view_578, [512, 768]);  view_578 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_579, permute_498);  permute_498 = None
    permute_499: "f32[768, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_499, view_22);  permute_499 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_501: "f32[768, 768]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_581: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_581);  add_175 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_582: "f32[512, 768]" = torch.ops.aten.view.default(view_573, [512, 768]);  view_573 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_582, permute_502);  permute_502 = None
    permute_503: "f32[768, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_503, view_22);  permute_503 = view_22 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_187, [768]);  sum_187 = None
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_584: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_176, view_584);  add_176 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, primals_24);  primals_24 = None
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, 768)
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True)
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_428, mul_8);  mul_428 = None
    sum_189: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True);  mul_430 = None
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_189);  sum_189 = None
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_429, sum_188);  mul_429 = sum_188 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_431);  sub_125 = mul_431 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_126);  div_60 = sub_126 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_177, mul_8);  mul_8 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 1]);  mul_433 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_177, [0, 1]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_432, mul_434);  mul_434 = None
    clone_67: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_435, memory_format = torch.contiguous_format);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[512, 768]" = torch.ops.aten.view.default(clone_67, [512, 768]);  clone_67 = None
    mm_136: "f32[512, 3072]" = torch.ops.aten.mm.default(view_585, permute_506);  permute_506 = None
    permute_507: "f32[768, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_507, view_20);  permute_507 = view_20 = None
    permute_508: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    permute_509: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_587: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_136, [1, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_437: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
    mul_438: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_439: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_438, -0.5);  mul_438 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_439);  mul_439 = None
    mul_440: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_441: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_440);  view_19 = mul_440 = None
    add_179: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_437, mul_441);  mul_437 = mul_441 = None
    mul_442: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_587, add_179);  view_587 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[512, 3072]" = torch.ops.aten.view.default(mul_442, [512, 3072]);  mul_442 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_588, permute_510);  permute_510 = None
    permute_511: "f32[3072, 512]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_511, view_18);  permute_511 = view_18 = None
    permute_512: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_193: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[3072]" = torch.ops.aten.view.default(sum_193, [3072]);  sum_193 = None
    permute_513: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_590: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_180: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_432, view_590);  mul_432 = view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_444: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_180, primals_18);  primals_18 = None
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_444, 768)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True)
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_444, mul_3);  mul_444 = None
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_195);  sum_195 = None
    sub_128: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_445, sum_194);  mul_445 = sum_194 = None
    sub_129: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_128, mul_447);  sub_128 = mul_447 = None
    mul_448: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_129);  div_61 = sub_129 = None
    mul_449: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_180, mul_3);  mul_3 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 1]);  mul_449 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_180, [0, 1]);  add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_450: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_451: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_448, mul_450);  mul_450 = None
    clone_68: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_451, memory_format = torch.contiguous_format);  mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_591: "f32[512, 768]" = torch.ops.aten.view.default(clone_68, [512, 768]);  clone_68 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_591, permute_514);  permute_514 = None
    permute_515: "f32[768, 512]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_16);  permute_515 = view_16 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.view.default(sum_198, [768]);  sum_198 = None
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_593: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_594: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_593, [1, 512, 12, 64]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_518: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_518, clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale = 0.125);  permute_518 = clone_default_33 = clone_default_34 = clone_default_35 = alias_default_23 = getitem_204 = getitem_205 = getitem_206 = None
    getitem_207: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
    getitem_208: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
    getitem_209: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_70, [1, 512, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_602: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_71, [1, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_603: "f32[512, 768]" = torch.ops.aten.view.default(view_602, [512, 768]);  view_602 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_603, permute_526);  permute_526 = None
    permute_527: "f32[768, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_527, view);  permute_527 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    permute_529: "f32[768, 768]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_605: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_181: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_448, view_605);  mul_448 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_530: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_606: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_530, [1, 512, 768]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_607: "f32[512, 768]" = torch.ops.aten.view.default(view_606, [512, 768]);  view_606 = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_607, permute_531);  permute_531 = None
    permute_532: "f32[768, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_532, view);  permute_532 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    permute_534: "f32[768, 768]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_609: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_182: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_181, view_609);  add_181 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_610: "f32[512, 768]" = torch.ops.aten.view.default(view_601, [512, 768]);  view_601 = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_610, permute_535);  permute_535 = None
    permute_536: "f32[768, 512]" = torch.ops.aten.permute.default(view_610, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_536, view);  permute_536 = view = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
    view_611: "f32[768]" = torch.ops.aten.view.default(sum_202, [768]);  sum_202 = None
    permute_538: "f32[768, 768]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_612: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_182, view_612);  add_182 = view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:127, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_456: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_457: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_183, mul_456);  add_183 = mul_456 = None
    clone_72: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_457, memory_format = torch.contiguous_format);  mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:126, code: embeddings = self.LayerNorm(embeddings)
    mul_459: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_72, primals_8);  primals_8 = None
    mul_460: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_459, 768)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_459, mul_1);  mul_459 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
    sub_132: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_460, sum_203);  mul_460 = sum_203 = None
    sub_133: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_462);  sub_132 = mul_462 = None
    mul_463: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_133);  div_63 = sub_133 = None
    mul_464: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_72, mul_1);  mul_1 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_72, [0, 1]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:113, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    full_default_9: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(full_default_9, full_default_5, mul_463);  full_default_9 = None
    full_default_11: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[2, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [full_default], where_4, True);  full_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:112, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
    full_default_14: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [full_default], where_4, True);  full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:107, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
    _unsafe_index_put_3: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [select_3], where_4, True);  select_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
    _unsafe_index_put_4: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [select_2], where_4, True);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    _unsafe_index_put_5: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [select_1], where_4, True);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    add_184: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_index_put_3, _unsafe_index_put_5);  _unsafe_index_put_3 = _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    _unsafe_index_put_6: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [select], where_4, True);  full_default_14 = select = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    add_185: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_index_put_4, _unsafe_index_put_6);  _unsafe_index_put_4 = _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:102, code: position_embeddings = self.position_embeddings(position_ids)
    eq_7: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_11: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
    where_11: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_11, full_default_5, mul_463);  unsqueeze_11 = None
    full_default_31: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_7: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_31, [slice_1], where_11, True);  full_default_31 = slice_1 = where_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_8: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_211, 0)
    unsqueeze_12: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_8, -1);  eq_8 = None
    where_12: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_12, full_default_5, mul_463);  unsqueeze_12 = full_default_5 = mul_463 = None
    full_default_33: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_8: "f32[30522, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_33, [primals_211], where_12, True);  full_default_33 = primals_211 = where_12 = None
    return [_unsafe_index_put_8, _unsafe_index_put_7, add_185, add_184, _unsafe_index_put_1, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_538, view_611, permute_534, view_608, permute_529, view_604, permute_517, view_592, sum_196, sum_197, permute_513, view_589, permute_509, view_586, sum_190, sum_191, permute_505, view_583, permute_501, view_580, permute_496, view_576, permute_484, view_564, sum_181, sum_182, permute_480, view_561, permute_476, view_558, sum_175, sum_176, permute_472, view_555, permute_468, view_552, permute_463, view_548, permute_451, view_536, sum_166, sum_167, permute_447, view_533, permute_443, view_530, sum_160, sum_161, permute_439, view_527, permute_435, view_524, permute_430, view_520, permute_418, view_508, sum_151, sum_152, permute_414, view_505, permute_410, view_502, sum_145, sum_146, permute_406, view_499, permute_402, view_496, permute_397, view_492, permute_385, view_480, sum_136, sum_137, permute_381, view_477, permute_377, view_474, sum_130, sum_131, permute_373, view_471, permute_369, view_468, permute_364, view_464, permute_352, view_452, sum_121, sum_122, permute_348, view_449, permute_344, view_446, sum_115, sum_116, permute_340, view_443, permute_336, view_440, permute_331, view_436, permute_319, view_424, sum_106, sum_107, permute_315, view_421, permute_311, view_418, sum_100, sum_101, permute_307, view_415, permute_303, view_412, permute_298, view_408, permute_286, view_396, sum_91, sum_92, permute_282, view_393, permute_278, view_390, sum_85, sum_86, permute_274, view_387, permute_270, view_384, permute_265, view_380, permute_253, view_368, sum_76, sum_77, permute_249, view_365, permute_245, view_362, sum_70, sum_71, permute_241, view_359, permute_237, view_356, permute_232, view_352, permute_220, view_340, sum_61, sum_62, permute_216, view_337, permute_212, view_334, sum_55, sum_56, permute_208, view_331, permute_204, view_328, permute_199, view_324, permute_187, view_312, sum_46, sum_47, permute_183, view_309, permute_179, view_306, sum_40, sum_41, permute_175, view_303, permute_171, view_300, permute_166, view_296, permute_154, view_284, sum_31, sum_32, permute_150, view_281, permute_146, view_278, sum_25, sum_26, None, None, permute_142, view_275, sum_20, sum_21, permute_138, view_272, None, None, None]
    