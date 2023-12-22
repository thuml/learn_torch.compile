from __future__ import annotations



def forward(self, primals_8: "f32[768]", primals_18: "f32[768]", primals_24: "f32[768]", primals_34: "f32[768]", primals_40: "f32[768]", primals_50: "f32[768]", primals_56: "f32[768]", primals_66: "f32[768]", primals_72: "f32[768]", primals_82: "f32[768]", primals_88: "f32[768]", primals_98: "f32[768]", primals_104: "f32[768]", primals_114: "f32[768]", primals_120: "f32[768]", primals_130: "f32[768]", primals_136: "f32[768]", primals_146: "f32[768]", primals_152: "f32[768]", primals_162: "f32[768]", primals_168: "f32[768]", primals_178: "f32[768]", primals_184: "f32[768]", primals_194: "f32[768]", primals_200: "f32[768]", primals_207: "i64[1, 512]", full_default: "i64[1, 512]", slice_1: "i64[1, 512]", select: "i64[1, 512]", select_1: "i64[1, 512]", select_2: "i64[1, 512]", select_3: "i64[1, 512]", mul_1: "f32[1, 512, 768]", getitem_3: "b8[1, 512, 768]", view: "f32[512, 768]", clone_default_33: "f32[1, 12, 512, 64]", clone_default_34: "f32[1, 12, 512, 64]", clone_default_35: "f32[1, 12, 512, 64]", getitem_204: "f32[1, 12, 512]", getitem_205: "i64[]", getitem_206: "i64[]", alias_default_23: "f32[1, 12, 512, 64]", view_16: "f32[512, 768]", getitem_7: "b8[1, 512, 768]", mul_3: "f32[1, 512, 768]", view_18: "f32[512, 768]", addmm_4: "f32[512, 3072]", view_20: "f32[512, 3072]", getitem_11: "b8[1, 512, 768]", mul_8: "f32[1, 512, 768]", view_22: "f32[512, 768]", clone_default_30: "f32[1, 12, 512, 64]", clone_default_31: "f32[1, 12, 512, 64]", clone_default_32: "f32[1, 12, 512, 64]", getitem_197: "f32[1, 12, 512]", getitem_198: "i64[]", getitem_199: "i64[]", alias_default_21: "f32[1, 12, 512, 64]", view_38: "f32[512, 768]", getitem_17: "b8[1, 512, 768]", mul_10: "f32[1, 512, 768]", view_40: "f32[512, 768]", addmm_10: "f32[512, 3072]", view_42: "f32[512, 3072]", getitem_21: "b8[1, 512, 768]", mul_15: "f32[1, 512, 768]", view_44: "f32[512, 768]", clone_default_27: "f32[1, 12, 512, 64]", clone_default_28: "f32[1, 12, 512, 64]", clone_default_29: "f32[1, 12, 512, 64]", getitem_190: "f32[1, 12, 512]", getitem_191: "i64[]", getitem_192: "i64[]", alias_default_19: "f32[1, 12, 512, 64]", view_60: "f32[512, 768]", getitem_27: "b8[1, 512, 768]", mul_17: "f32[1, 512, 768]", view_62: "f32[512, 768]", addmm_16: "f32[512, 3072]", view_64: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_22: "f32[1, 512, 768]", view_66: "f32[512, 768]", clone_default_24: "f32[1, 12, 512, 64]", clone_default_25: "f32[1, 12, 512, 64]", clone_default_26: "f32[1, 12, 512, 64]", getitem_183: "f32[1, 12, 512]", getitem_184: "i64[]", getitem_185: "i64[]", alias_default_17: "f32[1, 12, 512, 64]", view_82: "f32[512, 768]", getitem_37: "b8[1, 512, 768]", mul_24: "f32[1, 512, 768]", view_84: "f32[512, 768]", addmm_22: "f32[512, 3072]", view_86: "f32[512, 3072]", getitem_41: "b8[1, 512, 768]", mul_29: "f32[1, 512, 768]", view_88: "f32[512, 768]", clone_default_21: "f32[1, 12, 512, 64]", clone_default_22: "f32[1, 12, 512, 64]", clone_default_23: "f32[1, 12, 512, 64]", getitem_176: "f32[1, 12, 512]", getitem_177: "i64[]", getitem_178: "i64[]", alias_default_15: "f32[1, 12, 512, 64]", view_104: "f32[512, 768]", getitem_47: "b8[1, 512, 768]", mul_31: "f32[1, 512, 768]", view_106: "f32[512, 768]", addmm_28: "f32[512, 3072]", view_108: "f32[512, 3072]", getitem_51: "b8[1, 512, 768]", mul_36: "f32[1, 512, 768]", view_110: "f32[512, 768]", clone_default_18: "f32[1, 12, 512, 64]", clone_default_19: "f32[1, 12, 512, 64]", clone_default_20: "f32[1, 12, 512, 64]", getitem_169: "f32[1, 12, 512]", getitem_170: "i64[]", getitem_171: "i64[]", alias_default_13: "f32[1, 12, 512, 64]", view_126: "f32[512, 768]", getitem_57: "b8[1, 512, 768]", mul_38: "f32[1, 512, 768]", view_128: "f32[512, 768]", addmm_34: "f32[512, 3072]", view_130: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_43: "f32[1, 512, 768]", view_132: "f32[512, 768]", clone_default_15: "f32[1, 12, 512, 64]", clone_default_16: "f32[1, 12, 512, 64]", clone_default_17: "f32[1, 12, 512, 64]", getitem_162: "f32[1, 12, 512]", getitem_163: "i64[]", getitem_164: "i64[]", alias_default_11: "f32[1, 12, 512, 64]", view_148: "f32[512, 768]", getitem_67: "b8[1, 512, 768]", mul_45: "f32[1, 512, 768]", view_150: "f32[512, 768]", addmm_40: "f32[512, 3072]", view_152: "f32[512, 3072]", getitem_71: "b8[1, 512, 768]", mul_50: "f32[1, 512, 768]", view_154: "f32[512, 768]", clone_default_12: "f32[1, 12, 512, 64]", clone_default_13: "f32[1, 12, 512, 64]", clone_default_14: "f32[1, 12, 512, 64]", getitem_155: "f32[1, 12, 512]", getitem_156: "i64[]", getitem_157: "i64[]", alias_default_9: "f32[1, 12, 512, 64]", view_170: "f32[512, 768]", getitem_77: "b8[1, 512, 768]", mul_52: "f32[1, 512, 768]", view_172: "f32[512, 768]", addmm_46: "f32[512, 3072]", view_174: "f32[512, 3072]", getitem_81: "b8[1, 512, 768]", mul_57: "f32[1, 512, 768]", view_176: "f32[512, 768]", clone_default_9: "f32[1, 12, 512, 64]", clone_default_10: "f32[1, 12, 512, 64]", clone_default_11: "f32[1, 12, 512, 64]", getitem_148: "f32[1, 12, 512]", getitem_149: "i64[]", getitem_150: "i64[]", alias_default_7: "f32[1, 12, 512, 64]", view_192: "f32[512, 768]", getitem_87: "b8[1, 512, 768]", mul_59: "f32[1, 512, 768]", view_194: "f32[512, 768]", addmm_52: "f32[512, 3072]", view_196: "f32[512, 3072]", getitem_91: "b8[1, 512, 768]", mul_64: "f32[1, 512, 768]", view_198: "f32[512, 768]", clone_default_6: "f32[1, 12, 512, 64]", clone_default_7: "f32[1, 12, 512, 64]", clone_default_8: "f32[1, 12, 512, 64]", getitem_141: "f32[1, 12, 512]", getitem_142: "i64[]", getitem_143: "i64[]", alias_default_5: "f32[1, 12, 512, 64]", view_214: "f32[512, 768]", getitem_97: "b8[1, 512, 768]", mul_66: "f32[1, 512, 768]", view_216: "f32[512, 768]", addmm_58: "f32[512, 3072]", view_218: "f32[512, 3072]", getitem_101: "b8[1, 512, 768]", mul_71: "f32[1, 512, 768]", view_220: "f32[512, 768]", clone_default_3: "f32[1, 12, 512, 64]", clone_default_4: "f32[1, 12, 512, 64]", clone_default_5: "f32[1, 12, 512, 64]", getitem_134: "f32[1, 12, 512]", getitem_135: "i64[]", getitem_136: "i64[]", alias_default_3: "f32[1, 12, 512, 64]", view_236: "f32[512, 768]", getitem_107: "b8[1, 512, 768]", mul_73: "f32[1, 512, 768]", view_238: "f32[512, 768]", addmm_64: "f32[512, 3072]", view_240: "f32[512, 3072]", getitem_111: "b8[1, 512, 768]", mul_78: "f32[1, 512, 768]", view_242: "f32[512, 768]", clone_default: "f32[1, 12, 512, 64]", clone_default_1: "f32[1, 12, 512, 64]", clone_default_2: "f32[1, 12, 512, 64]", getitem_127: "f32[1, 12, 512]", getitem_128: "i64[]", getitem_129: "i64[]", alias_default_1: "f32[1, 12, 512, 64]", view_258: "f32[512, 768]", getitem_117: "b8[1, 512, 768]", mul_80: "f32[1, 512, 768]", view_260: "f32[512, 768]", addmm_70: "f32[512, 3072]", view_262: "f32[512, 3072]", getitem_121: "b8[1, 512, 768]", mul_85: "f32[1, 512, 768]", select_8: "f32[1, 768]", tanh: "f32[1, 768]", getitem_124: "f32[1, 768]", getitem_125: "b8[1, 768]", permute_134: "f32[2, 768]", permute_138: "f32[768, 768]", div_24: "f32[1, 512, 1]", permute_142: "f32[768, 3072]", permute_146: "f32[3072, 768]", div_25: "f32[1, 512, 1]", permute_150: "f32[768, 768]", permute_162: "f32[768, 768]", permute_167: "f32[768, 768]", permute_171: "f32[768, 768]", div_27: "f32[1, 512, 1]", permute_175: "f32[768, 3072]", permute_179: "f32[3072, 768]", div_28: "f32[1, 512, 1]", permute_183: "f32[768, 768]", permute_195: "f32[768, 768]", permute_200: "f32[768, 768]", permute_204: "f32[768, 768]", div_30: "f32[1, 512, 1]", permute_208: "f32[768, 3072]", permute_212: "f32[3072, 768]", div_31: "f32[1, 512, 1]", permute_216: "f32[768, 768]", permute_228: "f32[768, 768]", permute_233: "f32[768, 768]", permute_237: "f32[768, 768]", div_33: "f32[1, 512, 1]", permute_241: "f32[768, 3072]", permute_245: "f32[3072, 768]", div_34: "f32[1, 512, 1]", permute_249: "f32[768, 768]", permute_261: "f32[768, 768]", permute_266: "f32[768, 768]", permute_270: "f32[768, 768]", div_36: "f32[1, 512, 1]", permute_274: "f32[768, 3072]", permute_278: "f32[3072, 768]", div_37: "f32[1, 512, 1]", permute_282: "f32[768, 768]", permute_294: "f32[768, 768]", permute_299: "f32[768, 768]", permute_303: "f32[768, 768]", div_39: "f32[1, 512, 1]", permute_307: "f32[768, 3072]", permute_311: "f32[3072, 768]", div_40: "f32[1, 512, 1]", permute_315: "f32[768, 768]", permute_327: "f32[768, 768]", permute_332: "f32[768, 768]", permute_336: "f32[768, 768]", div_42: "f32[1, 512, 1]", permute_340: "f32[768, 3072]", permute_344: "f32[3072, 768]", div_43: "f32[1, 512, 1]", permute_348: "f32[768, 768]", permute_360: "f32[768, 768]", permute_365: "f32[768, 768]", permute_369: "f32[768, 768]", div_45: "f32[1, 512, 1]", permute_373: "f32[768, 3072]", permute_377: "f32[3072, 768]", div_46: "f32[1, 512, 1]", permute_381: "f32[768, 768]", permute_393: "f32[768, 768]", permute_398: "f32[768, 768]", permute_402: "f32[768, 768]", div_48: "f32[1, 512, 1]", permute_406: "f32[768, 3072]", permute_410: "f32[3072, 768]", div_49: "f32[1, 512, 1]", permute_414: "f32[768, 768]", permute_426: "f32[768, 768]", permute_431: "f32[768, 768]", permute_435: "f32[768, 768]", div_51: "f32[1, 512, 1]", permute_439: "f32[768, 3072]", permute_443: "f32[3072, 768]", div_52: "f32[1, 512, 1]", permute_447: "f32[768, 768]", permute_459: "f32[768, 768]", permute_464: "f32[768, 768]", permute_468: "f32[768, 768]", div_54: "f32[1, 512, 1]", permute_472: "f32[768, 3072]", permute_476: "f32[3072, 768]", div_55: "f32[1, 512, 1]", permute_480: "f32[768, 768]", permute_492: "f32[768, 768]", permute_497: "f32[768, 768]", permute_501: "f32[768, 768]", div_57: "f32[1, 512, 1]", permute_505: "f32[768, 3072]", permute_509: "f32[3072, 768]", div_58: "f32[1, 512, 1]", permute_513: "f32[768, 768]", permute_525: "f32[768, 768]", permute_530: "f32[768, 768]", permute_534: "f32[768, 768]", div_60: "f32[1, 512, 1]", tangents_1: "f32[1, 512, 768]", tangents_2: "f32[1, 768]", tangents_3: "f32[1, 2]"):
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:560, code: pooled_output = self.activation(pooled_output)
    alias_12: "f32[1, 768]" = torch.ops.aten.alias.default(tanh);  tanh = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1085, code: logits = self.classifier(pooled_output)
    mm: "f32[1, 768]" = torch.ops.aten.mm.default(tangents_3, permute_134);  permute_134 = None
    permute_135: "f32[2, 1]" = torch.ops.aten.permute.default(tangents_3, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_135, getitem_124);  permute_135 = getitem_124 = None
    permute_136: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(tangents_3, [0], True);  tangents_3 = None
    view_264: "f32[2]" = torch.ops.aten.view.default(sum_13, [2]);  sum_13 = None
    permute_137: "f32[2, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1084, code: pooled_output = self.dropout(pooled_output)
    convert_element_type: "f32[1, 768]" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
    mul_87: "f32[1, 768]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
    mul_88: "f32[1, 768]" = torch.ops.aten.mul.Tensor(mm, mul_87);  mm = mul_87 = None
    clone_12: "f32[1, 768]" = torch.ops.aten.clone.default(mul_88, memory_format = torch.contiguous_format);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:1084, code: pooled_output = self.dropout(pooled_output)
    add_106: "f32[1, 768]" = torch.ops.aten.add.Tensor(tangents_2, clone_12);  tangents_2 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:560, code: pooled_output = self.activation(pooled_output)
    alias_13: "f32[1, 768]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_89: "f32[1, 768]" = torch.ops.aten.mul.Tensor(alias_13, alias_13);  alias_13 = None
    sub_40: "f32[1, 768]" = torch.ops.aten.sub.Tensor(1, mul_89);  mul_89 = None
    mul_90: "f32[1, 768]" = torch.ops.aten.mul.Tensor(add_106, sub_40);  add_106 = sub_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:559, code: pooled_output = self.dense(first_token_tensor)
    mm_2: "f32[1, 768]" = torch.ops.aten.mm.default(mul_90, permute_138);  permute_138 = None
    permute_139: "f32[768, 1]" = torch.ops.aten.permute.default(mul_90, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, select_8);  permute_139 = select_8 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_14: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(mul_90, [0], True);  mul_90 = None
    view_265: "f32[768]" = torch.ops.aten.view.default(sum_14, [768]);  sum_14 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:558, code: first_token_tensor = hidden_states[:, 0]
    full_default_4: "f32[1, 512, 768]" = torch.ops.aten.full.default([1, 512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[1, 512, 768]" = torch.ops.aten.select_scatter.default(full_default_4, mm_2, 1, 0);  mm_2 = None
    slice_scatter: "f32[1, 512, 768]" = torch.ops.aten.slice_scatter.default(full_default_4, select_scatter, 0, 0, 9223372036854775807);  full_default_4 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:558, code: first_token_tensor = hidden_states[:, 0]
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(tangents_1, slice_scatter);  tangents_1 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, primals_200);  primals_200 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, 768)
    sum_15: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True)
    mul_94: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, mul_85);  mul_92 = None
    sum_16: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True);  mul_94 = None
    mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_16);  sum_16 = None
    sub_42: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_93, sum_15);  mul_93 = sum_15 = None
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_42, mul_95);  sub_42 = mul_95 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_43);  div_24 = sub_43 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, mul_85);  mul_85 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_97, [0, 1]);  mul_97 = None
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_107, [0, 1]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, mul_98);  mul_98 = None
    clone_13: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_99, memory_format = torch.contiguous_format);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_266: "f32[512, 768]" = torch.ops.aten.view.default(clone_13, [512, 768]);  clone_13 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_266, permute_142);  permute_142 = None
    permute_143: "f32[768, 512]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_268: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_102: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_103: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_102, -0.5);  mul_102 = None
    exp_12: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_103);  mul_103 = None
    mul_104: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_105: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_104);  view_261 = mul_104 = None
    add_109: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_101, mul_105);  mul_101 = mul_105 = None
    mul_106: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_268, add_109);  view_268 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_269: "f32[512, 3072]" = torch.ops.aten.view.default(mul_106, [512, 3072]);  mul_106 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_269, permute_146);  permute_146 = None
    permute_147: "f32[3072, 512]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[3072]" = torch.ops.aten.view.default(sum_20, [3072]);  sum_20 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_271: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_96, view_271);  mul_96 = view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_194);  primals_194 = None
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_21: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_80);  mul_108 = None
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_22);  sum_22 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_21);  mul_109 = sum_21 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_111);  sub_45 = mul_111 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_46);  div_25 = sub_46 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_80);  mul_80 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_112, mul_114);  mul_114 = None
    clone_14: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_115, memory_format = torch.contiguous_format);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[512, 768]" = torch.ops.aten.view.default(clone_14, [512, 768]);  clone_14 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_272, permute_150);  permute_150 = None
    permute_151: "f32[768, 512]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
    permute_152: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_274: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_275: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_274, [1, 512, 12, 64]);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_154: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_154, clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_154 = clone_default = clone_default_1 = clone_default_2 = alias_default_1 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_130: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
    getitem_131: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
    getitem_132: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_160: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_282: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_16, [1, 512, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_283: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_17, [1, 512, 768]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_284: "f32[512, 768]" = torch.ops.aten.view.default(view_283, [512, 768]);  view_283 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_284, permute_162);  permute_162 = None
    permute_163: "f32[768, 512]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_286: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_112, view_286);  mul_112 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_287: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_166, [1, 512, 768]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_288: "f32[512, 768]" = torch.ops.aten.view.default(view_287, [512, 768]);  view_287 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_288, permute_167);  permute_167 = None
    permute_168: "f32[768, 512]" = torch.ops.aten.permute.default(view_288, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_168, view_242);  permute_168 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
    view_289: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_290: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_112: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_111, view_290);  add_111 = view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_291: "f32[512, 768]" = torch.ops.aten.view.default(view_282, [512, 768]);  view_282 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_291, permute_171);  permute_171 = None
    permute_172: "f32[768, 512]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_29, [768]);  sum_29 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_293);  add_112 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_184);  primals_184 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_78);  mul_121 = None
    sum_31: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_31);  sum_31 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_30);  mul_122 = sum_30 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_49, mul_124);  sub_49 = mul_124 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_50);  div_27 = sub_50 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_78);  mul_78 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    clone_18: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_294: "f32[512, 768]" = torch.ops.aten.view.default(clone_18, [512, 768]);  clone_18 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_294, permute_175);  permute_175 = None
    permute_176: "f32[768, 512]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_296: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_94, 0.5);  add_94 = None
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_13: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_133);  view_239 = mul_133 = None
    add_115: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_296, add_115);  view_296 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_297: "f32[512, 3072]" = torch.ops.aten.view.default(mul_135, [512, 3072]);  mul_135 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_297, permute_179);  permute_179 = None
    permute_180: "f32[3072, 512]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_35: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[3072]" = torch.ops.aten.view.default(sum_35, [3072]);  sum_35 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_299: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_299);  mul_125 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_178);  primals_178 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
    sum_36: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_73);  mul_137 = None
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_37);  sum_37 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_36);  mul_138 = sum_36 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_140);  sub_52 = mul_140 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_53);  div_28 = sub_53 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_73);  mul_73 = None
    sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    clone_19: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_144, memory_format = torch.contiguous_format);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_300: "f32[512, 768]" = torch.ops.aten.view.default(clone_19, [512, 768]);  clone_19 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_300, permute_183);  permute_183 = None
    permute_184: "f32[768, 512]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_302: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_303: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_302, [1, 512, 12, 64]);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_187: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_187, clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale = 0.125);  permute_187 = clone_default_3 = clone_default_4 = clone_default_5 = alias_default_3 = getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
    getitem_138: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
    getitem_139: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_193: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_310: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_21, [1, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_311: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_22, [1, 512, 768]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_312: "f32[512, 768]" = torch.ops.aten.view.default(view_311, [512, 768]);  view_311 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_312, permute_195);  permute_195 = None
    permute_196: "f32[768, 512]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True);  view_312 = None
    view_313: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_314: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_141, view_314);  mul_141 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_315: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_199, [1, 512, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_316: "f32[512, 768]" = torch.ops.aten.view.default(view_315, [512, 768]);  view_315 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_316, permute_200);  permute_200 = None
    permute_201: "f32[768, 512]" = torch.ops.aten.permute.default(view_316, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_201, view_220);  permute_201 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True);  view_316 = None
    view_317: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_318: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_117, view_318);  add_117 = view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_319: "f32[512, 768]" = torch.ops.aten.view.default(view_310, [512, 768]);  view_310 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_319, permute_204);  permute_204 = None
    permute_205: "f32[768, 512]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
    view_320: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_118, view_321);  add_118 = view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_168);  primals_168 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_71);  mul_150 = None
    sum_46: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, sum_46);  sum_46 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_45);  mul_151 = sum_45 = None
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_153);  sub_56 = mul_153 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_57);  div_30 = sub_57 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_71);  mul_71 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_156);  mul_156 = None
    clone_23: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_157, memory_format = torch.contiguous_format);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_322: "f32[512, 768]" = torch.ops.aten.view.default(clone_23, [512, 768]);  clone_23 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_322, permute_208);  permute_208 = None
    permute_209: "f32[768, 512]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_324: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
    exp_14: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_161);  mul_161 = None
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_162);  view_217 = mul_162 = None
    add_121: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_324, add_121);  view_324 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_325: "f32[512, 3072]" = torch.ops.aten.view.default(mul_164, [512, 3072]);  mul_164 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_325, permute_212);  permute_212 = None
    permute_213: "f32[3072, 512]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_50: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[3072]" = torch.ops.aten.view.default(sum_50, [3072]);  sum_50 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_327: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_154, view_327);  mul_154 = view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_162);  primals_162 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, 768)
    sum_51: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, mul_66);  mul_166 = None
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_52);  sum_52 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_167, sum_51);  mul_167 = sum_51 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_169);  sub_59 = mul_169 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_60);  div_31 = sub_60 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_66);  mul_66 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, mul_172);  mul_172 = None
    clone_24: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_173, memory_format = torch.contiguous_format);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 768]" = torch.ops.aten.view.default(clone_24, [512, 768]);  clone_24 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_328, permute_216);  permute_216 = None
    permute_217: "f32[768, 512]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_330: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_331: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_330, [1, 512, 12, 64]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_220: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_220, clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale = 0.125);  permute_220 = clone_default_6 = clone_default_7 = clone_default_8 = alias_default_5 = getitem_141 = getitem_142 = getitem_143 = None
    getitem_144: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
    getitem_145: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
    getitem_146: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_338: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_26, [1, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_339: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_27, [1, 512, 768]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_340: "f32[512, 768]" = torch.ops.aten.view.default(view_339, [512, 768]);  view_339 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_340, permute_228);  permute_228 = None
    permute_229: "f32[768, 512]" = torch.ops.aten.permute.default(view_340, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
    view_341: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_342: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_170, view_342);  mul_170 = view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_232: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_343: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_232, [1, 512, 768]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_344: "f32[512, 768]" = torch.ops.aten.view.default(view_343, [512, 768]);  view_343 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_344, permute_233);  permute_233 = None
    permute_234: "f32[768, 512]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_198);  permute_234 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_346: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_124: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_346);  add_123 = view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_347: "f32[512, 768]" = torch.ops.aten.view.default(view_338, [512, 768]);  view_338 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_347, permute_237);  permute_237 = None
    permute_238: "f32[768, 512]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[768]" = torch.ops.aten.view.default(sum_59, [768]);  sum_59 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_349);  add_124 = view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_152);  primals_152 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_64);  mul_179 = None
    sum_61: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_61);  sum_61 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_60);  mul_180 = sum_60 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_182);  sub_63 = mul_182 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_64);  div_33 = sub_64 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_64);  mul_64 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_63: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, mul_185);  mul_185 = None
    clone_28: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_186, memory_format = torch.contiguous_format);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_350: "f32[512, 768]" = torch.ops.aten.view.default(clone_28, [512, 768]);  clone_28 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_350, permute_241);  permute_241 = None
    permute_242: "f32[768, 512]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_352: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_191);  view_195 = mul_191 = None
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_352, add_127);  view_352 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_353: "f32[512, 3072]" = torch.ops.aten.view.default(mul_193, [512, 3072]);  mul_193 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_353, permute_245);  permute_245 = None
    permute_246: "f32[3072, 512]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_65: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[3072]" = torch.ops.aten.view.default(sum_65, [3072]);  sum_65 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_355: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_128: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_183, view_355);  mul_183 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_146);  primals_146 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, 768)
    sum_66: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, mul_59);  mul_195 = None
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_67);  sum_67 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_196, sum_66);  mul_196 = sum_66 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_198);  sub_66 = mul_198 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_67);  div_34 = sub_67 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_59);  mul_59 = None
    sum_68: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, mul_201);  mul_201 = None
    clone_29: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_202, memory_format = torch.contiguous_format);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(clone_29, [512, 768]);  clone_29 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_356, permute_249);  permute_249 = None
    permute_250: "f32[768, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.view.default(sum_70, [768]);  sum_70 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_358: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_359: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_358, [1, 512, 12, 64]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_253: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_253, clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale = 0.125);  permute_253 = clone_default_9 = clone_default_10 = clone_default_11 = alias_default_7 = getitem_148 = getitem_149 = getitem_150 = None
    getitem_151: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
    getitem_152: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
    getitem_153: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_259: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_31, [1, 512, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_367: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_32, [1, 512, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_368: "f32[512, 768]" = torch.ops.aten.view.default(view_367, [512, 768]);  view_367 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_368, permute_261);  permute_261 = None
    permute_262: "f32[768, 512]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_370: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_199, view_370);  mul_199 = view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_265: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_371: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_265, [1, 512, 768]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_372: "f32[512, 768]" = torch.ops.aten.view.default(view_371, [512, 768]);  view_371 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_372, permute_266);  permute_266 = None
    permute_267: "f32[768, 512]" = torch.ops.aten.permute.default(view_372, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_176);  permute_267 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_372, [0], True);  view_372 = None
    view_373: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_374: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_374);  add_129 = view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_375: "f32[512, 768]" = torch.ops.aten.view.default(view_366, [512, 768]);  view_366 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_375, permute_270);  permute_270 = None
    permute_271: "f32[768, 512]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_130, view_377);  add_130 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_136);  primals_136 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_57);  mul_208 = None
    sum_76: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_76);  sum_76 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_75);  mul_209 = sum_75 = None
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_70, mul_211);  sub_70 = mul_211 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_71);  div_36 = sub_71 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_57);  mul_57 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, mul_214);  mul_214 = None
    clone_33: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_215, memory_format = torch.contiguous_format);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_378: "f32[512, 768]" = torch.ops.aten.view.default(clone_33, [512, 768]);  clone_33 = None
    mm_52: "f32[512, 3072]" = torch.ops.aten.mm.default(view_378, permute_274);  permute_274 = None
    permute_275: "f32[768, 512]" = torch.ops.aten.permute.default(view_378, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_277: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_380: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_52, [1, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_70, 0.5);  add_70 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_219: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_221: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_220);  view_173 = mul_220 = None
    add_133: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_380, add_133);  view_380 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 3072]" = torch.ops.aten.view.default(mul_222, [512, 3072]);  mul_222 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_381, permute_278);  permute_278 = None
    permute_279: "f32[3072, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[3072]" = torch.ops.aten.view.default(sum_80, [3072]);  sum_80 = None
    permute_281: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_383: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_212, view_383);  mul_212 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_130);  primals_130 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, 768)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_52);  mul_224 = None
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_82);  sum_82 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_225, sum_81);  mul_225 = sum_81 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_227);  sub_73 = mul_227 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_74);  div_37 = sub_74 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_52);  mul_52 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None
    clone_34: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_231, memory_format = torch.contiguous_format);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[512, 768]" = torch.ops.aten.view.default(clone_34, [512, 768]);  clone_34 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_384, permute_282);  permute_282 = None
    permute_283: "f32[768, 512]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[768]" = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_386: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_387: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_386, [1, 512, 12, 64]);  view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_286, clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale = 0.125);  permute_286 = clone_default_12 = clone_default_13 = clone_default_14 = alias_default_9 = getitem_155 = getitem_156 = getitem_157 = None
    getitem_158: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
    getitem_159: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
    getitem_160: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_394: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_36, [1, 512, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_395: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_37, [1, 512, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_396: "f32[512, 768]" = torch.ops.aten.view.default(view_395, [512, 768]);  view_395 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_396, permute_294);  permute_294 = None
    permute_295: "f32[768, 512]" = torch.ops.aten.permute.default(view_396, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True);  view_396 = None
    view_397: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_398: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_228, view_398);  mul_228 = view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_399: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_298, [1, 512, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_400: "f32[512, 768]" = torch.ops.aten.view.default(view_399, [512, 768]);  view_399 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_400, permute_299);  permute_299 = None
    permute_300: "f32[768, 512]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_154);  permute_300 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_402: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_402);  add_135 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_403: "f32[512, 768]" = torch.ops.aten.view.default(view_394, [512, 768]);  view_394 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_403, permute_303);  permute_303 = None
    permute_304: "f32[768, 512]" = torch.ops.aten.permute.default(view_403, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_405);  add_136 = view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_120);  primals_120 = None
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_50);  mul_237 = None
    sum_91: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_91);  sum_91 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_90);  mul_238 = sum_90 = None
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_240);  sub_77 = mul_240 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_78);  div_39 = sub_78 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_50);  mul_50 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_93: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_243);  mul_243 = None
    clone_38: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_244, memory_format = torch.contiguous_format);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.view.default(clone_38, [512, 768]);  clone_38 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_406, permute_307);  permute_307 = None
    permute_308: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_408: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_246: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_247: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_248: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_247, -0.5);  mul_247 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_248);  mul_248 = None
    mul_249: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_250: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_249);  view_151 = mul_249 = None
    add_139: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_246, mul_250);  mul_246 = mul_250 = None
    mul_251: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_408, add_139);  view_408 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_409: "f32[512, 3072]" = torch.ops.aten.view.default(mul_251, [512, 3072]);  mul_251 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_409, permute_311);  permute_311 = None
    permute_312: "f32[3072, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_95: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[3072]" = torch.ops.aten.view.default(sum_95, [3072]);  sum_95 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_411: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_241, view_411);  mul_241 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_114);  primals_114 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_45);  mul_253 = None
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_97);  sum_97 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_96);  mul_254 = sum_96 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_256);  sub_80 = mul_256 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_81);  div_40 = sub_81 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_45);  mul_45 = None
    sum_98: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    clone_39: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 768]" = torch.ops.aten.view.default(clone_39, [512, 768]);  clone_39 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_412, permute_315);  permute_315 = None
    permute_316: "f32[768, 512]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
    view_413: "f32[768]" = torch.ops.aten.view.default(sum_100, [768]);  sum_100 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_414: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_415: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_414, [1, 512, 12, 64]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_415, [0, 2, 1, 3]);  view_415 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_319, clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale = 0.125);  permute_319 = clone_default_15 = clone_default_16 = clone_default_17 = alias_default_11 = getitem_162 = getitem_163 = getitem_164 = None
    getitem_165: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
    getitem_166: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
    getitem_167: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_422: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_41, [1, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_423: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_42, [1, 512, 768]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_424: "f32[512, 768]" = torch.ops.aten.view.default(view_423, [512, 768]);  view_423 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_424, permute_327);  permute_327 = None
    permute_328: "f32[768, 512]" = torch.ops.aten.permute.default(view_424, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True);  view_424 = None
    view_425: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_426: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_257, view_426);  mul_257 = view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_427: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_331, [1, 512, 768]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_428: "f32[512, 768]" = torch.ops.aten.view.default(view_427, [512, 768]);  view_427 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_428, permute_332);  permute_332 = None
    permute_333: "f32[768, 512]" = torch.ops.aten.permute.default(view_428, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_132);  permute_333 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True);  view_428 = None
    view_429: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_430: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_430);  add_141 = view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_431: "f32[512, 768]" = torch.ops.aten.view.default(view_422, [512, 768]);  view_422 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_431, permute_336);  permute_336 = None
    permute_337: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_433);  add_142 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_104);  primals_104 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, 768)
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, mul_43);  mul_266 = None
    sum_106: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, sum_106);  sum_106 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_267, sum_105);  mul_267 = sum_105 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_269);  sub_84 = mul_269 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_85);  div_42 = sub_85 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_43);  mul_43 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None
    clone_43: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_273, memory_format = torch.contiguous_format);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 768]" = torch.ops.aten.view.default(clone_43, [512, 768]);  clone_43 = None
    mm_76: "f32[512, 3072]" = torch.ops.aten.mm.default(view_434, permute_340);  permute_340 = None
    permute_341: "f32[768, 512]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    permute_343: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_436: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_76, [1, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_275: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_54, 0.5);  add_54 = None
    mul_276: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_277: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
    mul_278: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_279: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_278);  view_129 = mul_278 = None
    add_145: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
    mul_280: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_436, add_145);  view_436 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[512, 3072]" = torch.ops.aten.view.default(mul_280, [512, 3072]);  mul_280 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_437, permute_344);  permute_344 = None
    permute_345: "f32[3072, 512]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_110: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[3072]" = torch.ops.aten.view.default(sum_110, [3072]);  sum_110 = None
    permute_347: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_439: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_270, view_439);  mul_270 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_282: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_98);  primals_98 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, 768)
    sum_111: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, mul_38);  mul_282 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_112);  sum_112 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_283, sum_111);  mul_283 = sum_111 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_285);  sub_87 = mul_285 = None
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_88);  div_43 = sub_88 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_38);  mul_38 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None
    clone_44: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_289, memory_format = torch.contiguous_format);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_440: "f32[512, 768]" = torch.ops.aten.view.default(clone_44, [512, 768]);  clone_44 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_440, permute_348);  permute_348 = None
    permute_349: "f32[768, 512]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_442: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_443: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_442, [1, 512, 12, 64]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_352, clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale = 0.125);  permute_352 = clone_default_18 = clone_default_19 = clone_default_20 = alias_default_13 = getitem_169 = getitem_170 = getitem_171 = None
    getitem_172: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
    getitem_173: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
    getitem_174: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_46, [1, 512, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_451: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_47, [1, 512, 768]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_452: "f32[512, 768]" = torch.ops.aten.view.default(view_451, [512, 768]);  view_451 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_452, permute_360);  permute_360 = None
    permute_361: "f32[768, 512]" = torch.ops.aten.permute.default(view_452, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True);  view_452 = None
    view_453: "f32[768]" = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_454: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_286, view_454);  mul_286 = view_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_455: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_364, [1, 512, 768]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_456: "f32[512, 768]" = torch.ops.aten.view.default(view_455, [512, 768]);  view_455 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_456, permute_365);  permute_365 = None
    permute_366: "f32[768, 512]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_366, view_110);  permute_366 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[768]" = torch.ops.aten.view.default(sum_118, [768]);  sum_118 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_458: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_147, view_458);  add_147 = view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_459: "f32[512, 768]" = torch.ops.aten.view.default(view_450, [512, 768]);  view_450 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_369);  permute_369 = None
    permute_370: "f32[768, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_148, view_461);  add_148 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_88);  primals_88 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_295, 768)
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_295, mul_36);  mul_295 = None
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_121);  sum_121 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_296, sum_120);  mul_296 = sum_120 = None
    sub_92: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_298);  sub_91 = mul_298 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_92);  div_45 = sub_92 = None
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_36);  mul_36 = None
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_123: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, mul_301);  mul_301 = None
    clone_48: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_302, memory_format = torch.contiguous_format);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_462: "f32[512, 768]" = torch.ops.aten.view.default(clone_48, [512, 768]);  clone_48 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_462, permute_373);  permute_373 = None
    permute_374: "f32[768, 512]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_464: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_304: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_46, 0.5);  add_46 = None
    mul_305: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_306: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_307);  view_107 = mul_307 = None
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_464, add_151);  view_464 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_465: "f32[512, 3072]" = torch.ops.aten.view.default(mul_309, [512, 3072]);  mul_309 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_465, permute_377);  permute_377 = None
    permute_378: "f32[3072, 512]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_125: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[3072]" = torch.ops.aten.view.default(sum_125, [3072]);  sum_125 = None
    permute_380: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_467: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_299, view_467);  mul_299 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_311: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_82);  primals_82 = None
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_126: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_31);  mul_311 = None
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, sum_127);  sum_127 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_126);  mul_312 = sum_126 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_314);  sub_94 = mul_314 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_95);  div_46 = sub_95 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_31);  mul_31 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, mul_317);  mul_317 = None
    clone_49: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_318, memory_format = torch.contiguous_format);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[512, 768]" = torch.ops.aten.view.default(clone_49, [512, 768]);  clone_49 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_468, permute_381);  permute_381 = None
    permute_382: "f32[768, 512]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
    view_469: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_470: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_471: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_470, [1, 512, 12, 64]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_385, clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale = 0.125);  permute_385 = clone_default_21 = clone_default_22 = clone_default_23 = alias_default_15 = getitem_176 = getitem_177 = getitem_178 = None
    getitem_179: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
    getitem_180: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
    getitem_181: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_51, [1, 512, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_479: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_52, [1, 512, 768]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_480: "f32[512, 768]" = torch.ops.aten.view.default(view_479, [512, 768]);  view_479 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_480, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_480, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_480, [0], True);  view_480 = None
    view_481: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_482: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_315, view_482);  mul_315 = view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_483: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_397, [1, 512, 768]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_484: "f32[512, 768]" = torch.ops.aten.view.default(view_483, [512, 768]);  view_483 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_484, permute_398);  permute_398 = None
    permute_399: "f32[768, 512]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_88);  permute_399 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[768]" = torch.ops.aten.view.default(sum_133, [768]);  sum_133 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_153, view_486);  add_153 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_487: "f32[512, 768]" = torch.ops.aten.view.default(view_478, [512, 768]);  view_478 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_487, permute_402);  permute_402 = None
    permute_403: "f32[768, 512]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[768]" = torch.ops.aten.view.default(sum_134, [768]);  sum_134 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_489);  add_154 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_72);  primals_72 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, 768)
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, mul_29);  mul_324 = None
    sum_136: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_136);  sum_136 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_325, sum_135);  mul_325 = sum_135 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_327);  sub_98 = mul_327 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_99);  div_48 = sub_99 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_29);  mul_29 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, mul_330);  mul_330 = None
    clone_53: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_331, memory_format = torch.contiguous_format);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_490: "f32[512, 768]" = torch.ops.aten.view.default(clone_53, [512, 768]);  clone_53 = None
    mm_100: "f32[512, 3072]" = torch.ops.aten.mm.default(view_490, permute_406);  permute_406 = None
    permute_407: "f32[768, 512]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    permute_409: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_492: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_100, [1, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_336);  view_85 = mul_336 = None
    add_157: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_492, add_157);  view_492 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_493: "f32[512, 3072]" = torch.ops.aten.view.default(mul_338, [512, 3072]);  mul_338 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_493, permute_410);  permute_410 = None
    permute_411: "f32[3072, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[3072]" = torch.ops.aten.view.default(sum_140, [3072]);  sum_140 = None
    permute_413: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_495: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_158: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_328, view_495);  mul_328 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_66);  primals_66 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, 768)
    sum_141: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, mul_24);  mul_340 = None
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_142);  sum_142 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_341, sum_141);  mul_341 = sum_141 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_343);  sub_101 = mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_102);  div_49 = sub_102 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_24);  mul_24 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_344, mul_346);  mul_346 = None
    clone_54: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_347, memory_format = torch.contiguous_format);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_496: "f32[512, 768]" = torch.ops.aten.view.default(clone_54, [512, 768]);  clone_54 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_414);  permute_414 = None
    permute_415: "f32[768, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_145: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_145, [768]);  sum_145 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_498: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_499: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_498, [1, 512, 12, 64]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_418, clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale = 0.125);  permute_418 = clone_default_24 = clone_default_25 = clone_default_26 = alias_default_17 = getitem_183 = getitem_184 = getitem_185 = None
    getitem_186: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
    getitem_187: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
    getitem_188: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_506: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_56, [1, 512, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_507: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_57, [1, 512, 768]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_508: "f32[512, 768]" = torch.ops.aten.view.default(view_507, [512, 768]);  view_507 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_508, permute_426);  permute_426 = None
    permute_427: "f32[768, 512]" = torch.ops.aten.permute.default(view_508, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True);  view_508 = None
    view_509: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_510: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_344, view_510);  mul_344 = view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_511: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_430, [1, 512, 768]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_512: "f32[512, 768]" = torch.ops.aten.view.default(view_511, [512, 768]);  view_511 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_512, permute_431);  permute_431 = None
    permute_432: "f32[768, 512]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_66);  permute_432 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True);  view_512 = None
    view_513: "f32[768]" = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_514: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_160: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_514);  add_159 = view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_515: "f32[512, 768]" = torch.ops.aten.view.default(view_506, [512, 768]);  view_506 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_435);  permute_435 = None
    permute_436: "f32[768, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_149, [768]);  sum_149 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_160, view_517);  add_160 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_56);  primals_56 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, 768)
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, mul_22);  mul_353 = None
    sum_151: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_151);  sum_151 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_354, sum_150);  mul_354 = sum_150 = None
    sub_106: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_356);  sub_105 = mul_356 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_106);  div_51 = sub_106 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_22);  mul_22 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_153: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_359 = None
    clone_58: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_360, memory_format = torch.contiguous_format);  mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_518: "f32[512, 768]" = torch.ops.aten.view.default(clone_58, [512, 768]);  clone_58 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_518, permute_439);  permute_439 = None
    permute_440: "f32[768, 512]" = torch.ops.aten.permute.default(view_518, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
    view_519: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_442: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_520: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_30, 0.5);  add_30 = None
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_365);  view_63 = mul_365 = None
    add_163: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_520, add_163);  view_520 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_521: "f32[512, 3072]" = torch.ops.aten.view.default(mul_367, [512, 3072]);  mul_367 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_521, permute_443);  permute_443 = None
    permute_444: "f32[3072, 512]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_155: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[3072]" = torch.ops.aten.view.default(sum_155, [3072]);  sum_155 = None
    permute_446: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_523: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_357, view_523);  mul_357 = view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_50);  primals_50 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, 768)
    sum_156: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_17);  mul_369 = None
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_157);  sum_157 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_370, sum_156);  mul_370 = sum_156 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_372);  sub_108 = mul_372 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_109);  div_52 = sub_109 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_17);  mul_17 = None
    sum_158: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, mul_375);  mul_375 = None
    clone_59: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_376, memory_format = torch.contiguous_format);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 768]" = torch.ops.aten.view.default(clone_59, [512, 768]);  clone_59 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_524, permute_447);  permute_447 = None
    permute_448: "f32[768, 512]" = torch.ops.aten.permute.default(view_524, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
    view_525: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_526: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_527: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_526, [1, 512, 12, 64]);  view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_451, clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale = 0.125);  permute_451 = clone_default_27 = clone_default_28 = clone_default_29 = alias_default_19 = getitem_190 = getitem_191 = getitem_192 = None
    getitem_193: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
    getitem_194: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
    getitem_195: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_61, [1, 512, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_535: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_62, [1, 512, 768]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_536: "f32[512, 768]" = torch.ops.aten.view.default(view_535, [512, 768]);  view_535 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_536, permute_459);  permute_459 = None
    permute_460: "f32[768, 512]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_538: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_373, view_538);  mul_373 = view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_539: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_463, [1, 512, 768]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_540: "f32[512, 768]" = torch.ops.aten.view.default(view_539, [512, 768]);  view_539 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_540, permute_464);  permute_464 = None
    permute_465: "f32[768, 512]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_542: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_542);  add_165 = view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_543: "f32[512, 768]" = torch.ops.aten.view.default(view_534, [512, 768]);  view_534 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_543, permute_468);  permute_468 = None
    permute_469: "f32[768, 512]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[768]" = torch.ops.aten.view.default(sum_164, [768]);  sum_164 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_545);  add_166 = view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_382: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_40);  primals_40 = None
    mul_383: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_382, 768)
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
    mul_384: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_382, mul_15);  mul_382 = None
    sum_166: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_166);  sum_166 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_383, sum_165);  mul_383 = sum_165 = None
    sub_113: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_385);  sub_112 = mul_385 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_113);  div_54 = sub_113 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_15);  mul_15 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_388);  mul_388 = None
    clone_63: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_389, memory_format = torch.contiguous_format);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_546: "f32[512, 768]" = torch.ops.aten.view.default(clone_63, [512, 768]);  clone_63 = None
    mm_124: "f32[512, 3072]" = torch.ops.aten.mm.default(view_546, permute_472);  permute_472 = None
    permute_473: "f32[768, 512]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_169: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
    view_547: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    permute_475: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_548: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_124, [1, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_391: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_22, 0.5);  add_22 = None
    mul_392: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_393: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
    mul_394: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_395: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_394);  view_41 = mul_394 = None
    add_169: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
    mul_396: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_548, add_169);  view_548 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_549: "f32[512, 3072]" = torch.ops.aten.view.default(mul_396, [512, 3072]);  mul_396 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_549, permute_476);  permute_476 = None
    permute_477: "f32[3072, 512]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_170: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[3072]" = torch.ops.aten.view.default(sum_170, [3072]);  sum_170 = None
    permute_479: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_551: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_386, view_551);  mul_386 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_34);  primals_34 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, 768)
    sum_171: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, mul_10);  mul_398 = None
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_172);  sum_172 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_399, sum_171);  mul_399 = sum_171 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_401);  sub_115 = mul_401 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_116);  div_55 = sub_116 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_10);  mul_10 = None
    sum_173: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    clone_64: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_405, memory_format = torch.contiguous_format);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[512, 768]" = torch.ops.aten.view.default(clone_64, [512, 768]);  clone_64 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_552, permute_480);  permute_480 = None
    permute_481: "f32[768, 512]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_175: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[768]" = torch.ops.aten.view.default(sum_175, [768]);  sum_175 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_554: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_555: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_554, [1, 512, 12, 64]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_484, clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale = 0.125);  permute_484 = clone_default_30 = clone_default_31 = clone_default_32 = alias_default_21 = getitem_197 = getitem_198 = getitem_199 = None
    getitem_200: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
    getitem_201: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
    getitem_202: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_66, [1, 512, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_563: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_67, [1, 512, 768]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_564: "f32[512, 768]" = torch.ops.aten.view.default(view_563, [512, 768]);  view_563 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_564, permute_492);  permute_492 = None
    permute_493: "f32[768, 512]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[768]" = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_566: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_171: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_402, view_566);  mul_402 = view_566 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_567: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_496, [1, 512, 768]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_568: "f32[512, 768]" = torch.ops.aten.view.default(view_567, [512, 768]);  view_567 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_568, permute_497);  permute_497 = None
    permute_498: "f32[768, 512]" = torch.ops.aten.permute.default(view_568, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_498, view_22);  permute_498 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True);  view_568 = None
    view_569: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_570: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_171, view_570);  add_171 = view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_571: "f32[512, 768]" = torch.ops.aten.view.default(view_562, [512, 768]);  view_562 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_501);  permute_501 = None
    permute_502: "f32[768, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_573);  add_172 = view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:358, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_24);  primals_24 = None
    mul_412: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_411, 768)
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_411, mul_8);  mul_411 = None
    sum_181: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_181);  sum_181 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_412, sum_180);  mul_412 = sum_180 = None
    sub_120: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_414);  sub_119 = mul_414 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_120);  div_57 = sub_120 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_8);  mul_8 = None
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_183: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:357, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, mul_417);  mul_417 = None
    clone_68: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:356, code: hidden_states = self.dense(hidden_states)
    view_574: "f32[512, 768]" = torch.ops.aten.view.default(clone_68, [512, 768]);  clone_68 = None
    mm_136: "f32[512, 3072]" = torch.ops.aten.mm.default(view_574, permute_505);  permute_505 = None
    permute_506: "f32[768, 512]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_184: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    permute_508: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_576: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_136, [1, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_420: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_14, 0.5);  add_14 = None
    mul_421: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_422: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_424: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_423);  view_19 = mul_423 = None
    add_175: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_576, add_175);  view_576 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    view_577: "f32[512, 3072]" = torch.ops.aten.view.default(mul_425, [512, 3072]);  mul_425 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_577, permute_509);  permute_509 = None
    permute_510: "f32[3072, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_185: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[3072]" = torch.ops.aten.view.default(sum_185, [3072]);  sum_185 = None
    permute_512: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_579: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:342, code: hidden_states = self.dense(hidden_states)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_415, view_579);  mul_415 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:277, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_18);  primals_18 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, 768)
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, mul_3);  mul_427 = None
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_187);  sum_187 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_428, sum_186);  mul_428 = sum_186 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_430);  sub_122 = mul_430 = None
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_123);  div_58 = sub_123 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_3);  mul_3 = None
    sum_188: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:276, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None
    clone_69: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_434, memory_format = torch.contiguous_format);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:275, code: hidden_states = self.dense(hidden_states)
    view_580: "f32[512, 768]" = torch.ops.aten.view.default(clone_69, [512, 768]);  clone_69 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_580, permute_513);  permute_513 = None
    permute_514: "f32[768, 512]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
    permute_515: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_190: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
    view_581: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_582: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:257, code: context_layer = context_layer.view(new_context_layer_shape)
    view_583: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_582, [1, 512, 12, 64]);  view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:255, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_583, [0, 2, 1, 3]);  view_583 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_517, clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale = 0.125);  permute_517 = clone_default_33 = clone_default_34 = clone_default_35 = alias_default_23 = getitem_204 = getitem_205 = getitem_206 = None
    getitem_207: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
    getitem_208: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
    getitem_209: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_590: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_71, [1, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_591: "f32[1, 512, 768]" = torch.ops.aten.view.default(clone_72, [1, 512, 768]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_592: "f32[512, 768]" = torch.ops.aten.view.default(view_591, [512, 768]);  view_591 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_592, permute_525);  permute_525 = None
    permute_526: "f32[768, 512]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_594: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:197, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_431, view_594);  mul_431 = view_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:162, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:161, code: x = x.view(new_x_shape)
    view_595: "f32[1, 512, 768]" = torch.ops.aten.view.default(permute_529, [1, 512, 768]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_596: "f32[512, 768]" = torch.ops.aten.view.default(view_595, [512, 768]);  view_595 = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_596, permute_530);  permute_530 = None
    permute_531: "f32[768, 512]" = torch.ops.aten.permute.default(view_596, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_193: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_596, [0], True);  view_596 = None
    view_597: "f32[768]" = torch.ops.aten.view.default(sum_193, [768]);  sum_193 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_598: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:196, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_178: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_177, view_598);  add_177 = view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    view_599: "f32[512, 768]" = torch.ops.aten.view.default(view_590, [512, 768]);  view_590 = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_599, permute_534);  permute_534 = None
    permute_535: "f32[768, 512]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True);  view_599 = None
    view_600: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:174, code: mixed_query_layer = self.query(hidden_states)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_178, view_601);  add_178 = view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:127, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_439: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_440: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_439);  add_179 = mul_439 = None
    clone_73: "f32[1, 512, 768]" = torch.ops.aten.clone.default(mul_440, memory_format = torch.contiguous_format);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:126, code: embeddings = self.LayerNorm(embeddings)
    mul_442: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_73, primals_8);  primals_8 = None
    mul_443: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, 768)
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True)
    mul_444: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, mul_1);  mul_442 = None
    sum_196: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_196);  sum_196 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_443, sum_195);  mul_443 = sum_195 = None
    sub_127: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_126, mul_445);  sub_126 = mul_445 = None
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_127);  div_60 = sub_127 = None
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(clone_73, mul_1);  mul_1 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
    sum_198: "f32[768]" = torch.ops.aten.sum.dim_IntList(clone_73, [0, 1]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:113, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    full_default_6: "b8[1, 512, 1]" = torch.ops.aten.full.default([1, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[1, 512, 768]" = torch.ops.aten.where.self(full_default_6, full_default_7, mul_446);  full_default_6 = None
    full_default_8: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[2, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_8, [full_default], where, True);  full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:112, code: w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
    full_default_11: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [full_default], where, True);  full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:107, code: lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
    _unsafe_index_put_3: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [select_3], where, True);  select_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:106, code: right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
    _unsafe_index_put_4: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [select_2], where, True);  select_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    _unsafe_index_put_5: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [select_1], where, True);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:105, code: upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
    add_180: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_index_put_3, _unsafe_index_put_5);  _unsafe_index_put_3 = _unsafe_index_put_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    _unsafe_index_put_6: "f32[1024, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_11, [select], where, True);  full_default_11 = select = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:104, code: left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
    add_181: "f32[1024, 768]" = torch.ops.aten.add.Tensor(_unsafe_index_put_4, _unsafe_index_put_6);  _unsafe_index_put_4 = _unsafe_index_put_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:102, code: position_embeddings = self.position_embeddings(position_ids)
    eq_7: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_7, -1);  eq_7 = None
    where_7: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_9, full_default_7, mul_446);  unsqueeze_9 = None
    full_default_28: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_7: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_28, [slice_1], where_7, True);  full_default_28 = slice_1 = where_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/layoutlm/modeling_layoutlm.py:99, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_8: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_207, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_8, -1);  eq_8 = None
    where_8: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_10, full_default_7, mul_446);  unsqueeze_10 = full_default_7 = mul_446 = None
    full_default_30: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_8: "f32[30522, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_30, [primals_207], where_8, True);  full_default_30 = primals_207 = where_8 = None
    return [_unsafe_index_put_8, _unsafe_index_put_7, add_181, add_180, _unsafe_index_put_1, _unsafe_index_put_1, _unsafe_index_put, sum_197, sum_198, permute_537, view_600, permute_533, view_597, permute_528, view_593, permute_516, view_581, sum_188, sum_189, permute_512, view_578, permute_508, view_575, sum_182, sum_183, permute_504, view_572, permute_500, view_569, permute_495, view_565, permute_483, view_553, sum_173, sum_174, permute_479, view_550, permute_475, view_547, sum_167, sum_168, permute_471, view_544, permute_467, view_541, permute_462, view_537, permute_450, view_525, sum_158, sum_159, permute_446, view_522, permute_442, view_519, sum_152, sum_153, permute_438, view_516, permute_434, view_513, permute_429, view_509, permute_417, view_497, sum_143, sum_144, permute_413, view_494, permute_409, view_491, sum_137, sum_138, permute_405, view_488, permute_401, view_485, permute_396, view_481, permute_384, view_469, sum_128, sum_129, permute_380, view_466, permute_376, view_463, sum_122, sum_123, permute_372, view_460, permute_368, view_457, permute_363, view_453, permute_351, view_441, sum_113, sum_114, permute_347, view_438, permute_343, view_435, sum_107, sum_108, permute_339, view_432, permute_335, view_429, permute_330, view_425, permute_318, view_413, sum_98, sum_99, permute_314, view_410, permute_310, view_407, sum_92, sum_93, permute_306, view_404, permute_302, view_401, permute_297, view_397, permute_285, view_385, sum_83, sum_84, permute_281, view_382, permute_277, view_379, sum_77, sum_78, permute_273, view_376, permute_269, view_373, permute_264, view_369, permute_252, view_357, sum_68, sum_69, permute_248, view_354, permute_244, view_351, sum_62, sum_63, permute_240, view_348, permute_236, view_345, permute_231, view_341, permute_219, view_329, sum_53, sum_54, permute_215, view_326, permute_211, view_323, sum_47, sum_48, permute_207, view_320, permute_203, view_317, permute_198, view_313, permute_186, view_301, sum_38, sum_39, permute_182, view_298, permute_178, view_295, sum_32, sum_33, permute_174, view_292, permute_170, view_289, permute_165, view_285, permute_153, view_273, sum_23, sum_24, permute_149, view_270, permute_145, view_267, sum_17, sum_18, permute_141, view_265, permute_137, view_264, None, None]
    