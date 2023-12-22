from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_14: "f32[768]", primals_20: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_46: "f32[768]", primals_52: "f32[768]", primals_62: "f32[768]", primals_68: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_94: "f32[768]", primals_100: "f32[768]", primals_110: "f32[768]", primals_116: "f32[768]", primals_126: "f32[768]", primals_132: "f32[768]", primals_142: "f32[768]", primals_148: "f32[768]", primals_158: "f32[768]", primals_164: "f32[768]", primals_174: "f32[768]", primals_180: "f32[768]", primals_190: "f32[768]", primals_196: "f32[768]", primals_202: "i64[1, 512]", expand: "i64[1, 512]", slice_4: "i64[1, 512]", mul_1: "f32[1, 512, 768]", getitem_3: "b8[1, 512, 768]", view: "f32[512, 768]", clone_default_33: "f32[1, 12, 512, 64]", clone_default_34: "f32[1, 12, 512, 64]", clone_default_35: "f32[1, 12, 512, 64]", getitem_204: "f32[1, 12, 512]", getitem_205: "i64[]", getitem_206: "i64[]", alias_default_23: "f32[1, 12, 512, 64]", view_16: "f32[512, 768]", getitem_7: "b8[1, 512, 768]", mul_3: "f32[1, 512, 768]", view_18: "f32[512, 768]", addmm_4: "f32[512, 3072]", view_20: "f32[512, 3072]", getitem_11: "b8[1, 512, 768]", mul_8: "f32[1, 512, 768]", view_22: "f32[512, 768]", clone_default_30: "f32[1, 12, 512, 64]", clone_default_31: "f32[1, 12, 512, 64]", clone_default_32: "f32[1, 12, 512, 64]", getitem_197: "f32[1, 12, 512]", getitem_198: "i64[]", getitem_199: "i64[]", alias_default_21: "f32[1, 12, 512, 64]", view_38: "f32[512, 768]", getitem_17: "b8[1, 512, 768]", mul_10: "f32[1, 512, 768]", view_40: "f32[512, 768]", addmm_10: "f32[512, 3072]", view_42: "f32[512, 3072]", getitem_21: "b8[1, 512, 768]", mul_15: "f32[1, 512, 768]", view_44: "f32[512, 768]", clone_default_27: "f32[1, 12, 512, 64]", clone_default_28: "f32[1, 12, 512, 64]", clone_default_29: "f32[1, 12, 512, 64]", getitem_190: "f32[1, 12, 512]", getitem_191: "i64[]", getitem_192: "i64[]", alias_default_19: "f32[1, 12, 512, 64]", view_60: "f32[512, 768]", getitem_27: "b8[1, 512, 768]", mul_17: "f32[1, 512, 768]", view_62: "f32[512, 768]", addmm_16: "f32[512, 3072]", view_64: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_22: "f32[1, 512, 768]", view_66: "f32[512, 768]", clone_default_24: "f32[1, 12, 512, 64]", clone_default_25: "f32[1, 12, 512, 64]", clone_default_26: "f32[1, 12, 512, 64]", getitem_183: "f32[1, 12, 512]", getitem_184: "i64[]", getitem_185: "i64[]", alias_default_17: "f32[1, 12, 512, 64]", view_82: "f32[512, 768]", getitem_37: "b8[1, 512, 768]", mul_24: "f32[1, 512, 768]", view_84: "f32[512, 768]", addmm_22: "f32[512, 3072]", view_86: "f32[512, 3072]", getitem_41: "b8[1, 512, 768]", mul_29: "f32[1, 512, 768]", view_88: "f32[512, 768]", clone_default_21: "f32[1, 12, 512, 64]", clone_default_22: "f32[1, 12, 512, 64]", clone_default_23: "f32[1, 12, 512, 64]", getitem_176: "f32[1, 12, 512]", getitem_177: "i64[]", getitem_178: "i64[]", alias_default_15: "f32[1, 12, 512, 64]", view_104: "f32[512, 768]", getitem_47: "b8[1, 512, 768]", mul_31: "f32[1, 512, 768]", view_106: "f32[512, 768]", addmm_28: "f32[512, 3072]", view_108: "f32[512, 3072]", getitem_51: "b8[1, 512, 768]", mul_36: "f32[1, 512, 768]", view_110: "f32[512, 768]", clone_default_18: "f32[1, 12, 512, 64]", clone_default_19: "f32[1, 12, 512, 64]", clone_default_20: "f32[1, 12, 512, 64]", getitem_169: "f32[1, 12, 512]", getitem_170: "i64[]", getitem_171: "i64[]", alias_default_13: "f32[1, 12, 512, 64]", view_126: "f32[512, 768]", getitem_57: "b8[1, 512, 768]", mul_38: "f32[1, 512, 768]", view_128: "f32[512, 768]", addmm_34: "f32[512, 3072]", view_130: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_43: "f32[1, 512, 768]", view_132: "f32[512, 768]", clone_default_15: "f32[1, 12, 512, 64]", clone_default_16: "f32[1, 12, 512, 64]", clone_default_17: "f32[1, 12, 512, 64]", getitem_162: "f32[1, 12, 512]", getitem_163: "i64[]", getitem_164: "i64[]", alias_default_11: "f32[1, 12, 512, 64]", view_148: "f32[512, 768]", getitem_67: "b8[1, 512, 768]", mul_45: "f32[1, 512, 768]", view_150: "f32[512, 768]", addmm_40: "f32[512, 3072]", view_152: "f32[512, 3072]", getitem_71: "b8[1, 512, 768]", mul_50: "f32[1, 512, 768]", view_154: "f32[512, 768]", clone_default_12: "f32[1, 12, 512, 64]", clone_default_13: "f32[1, 12, 512, 64]", clone_default_14: "f32[1, 12, 512, 64]", getitem_155: "f32[1, 12, 512]", getitem_156: "i64[]", getitem_157: "i64[]", alias_default_9: "f32[1, 12, 512, 64]", view_170: "f32[512, 768]", getitem_77: "b8[1, 512, 768]", mul_52: "f32[1, 512, 768]", view_172: "f32[512, 768]", addmm_46: "f32[512, 3072]", view_174: "f32[512, 3072]", getitem_81: "b8[1, 512, 768]", mul_57: "f32[1, 512, 768]", view_176: "f32[512, 768]", clone_default_9: "f32[1, 12, 512, 64]", clone_default_10: "f32[1, 12, 512, 64]", clone_default_11: "f32[1, 12, 512, 64]", getitem_148: "f32[1, 12, 512]", getitem_149: "i64[]", getitem_150: "i64[]", alias_default_7: "f32[1, 12, 512, 64]", view_192: "f32[512, 768]", getitem_87: "b8[1, 512, 768]", mul_59: "f32[1, 512, 768]", view_194: "f32[512, 768]", addmm_52: "f32[512, 3072]", view_196: "f32[512, 3072]", getitem_91: "b8[1, 512, 768]", mul_64: "f32[1, 512, 768]", view_198: "f32[512, 768]", clone_default_6: "f32[1, 12, 512, 64]", clone_default_7: "f32[1, 12, 512, 64]", clone_default_8: "f32[1, 12, 512, 64]", getitem_141: "f32[1, 12, 512]", getitem_142: "i64[]", getitem_143: "i64[]", alias_default_5: "f32[1, 12, 512, 64]", view_214: "f32[512, 768]", getitem_97: "b8[1, 512, 768]", mul_66: "f32[1, 512, 768]", view_216: "f32[512, 768]", addmm_58: "f32[512, 3072]", view_218: "f32[512, 3072]", getitem_101: "b8[1, 512, 768]", mul_71: "f32[1, 512, 768]", view_220: "f32[512, 768]", clone_default_3: "f32[1, 12, 512, 64]", clone_default_4: "f32[1, 12, 512, 64]", clone_default_5: "f32[1, 12, 512, 64]", getitem_134: "f32[1, 12, 512]", getitem_135: "i64[]", getitem_136: "i64[]", alias_default_3: "f32[1, 12, 512, 64]", view_236: "f32[512, 768]", getitem_107: "b8[1, 512, 768]", mul_73: "f32[1, 512, 768]", view_238: "f32[512, 768]", addmm_64: "f32[512, 3072]", view_240: "f32[512, 3072]", getitem_111: "b8[1, 512, 768]", mul_78: "f32[1, 512, 768]", view_242: "f32[512, 768]", clone_default: "f32[1, 12, 512, 64]", clone_default_1: "f32[1, 12, 512, 64]", clone_default_2: "f32[1, 12, 512, 64]", getitem_127: "f32[1, 12, 512]", getitem_128: "i64[]", getitem_129: "i64[]", alias_default_1: "f32[1, 12, 512, 64]", view_258: "f32[512, 768]", getitem_117: "b8[1, 512, 768]", mul_80: "f32[1, 512, 768]", view_260: "f32[512, 768]", addmm_70: "f32[512, 3072]", view_262: "f32[512, 3072]", getitem_121: "b8[1, 512, 768]", mul_85: "f32[1, 512, 768]", view_264: "f32[512, 768]", sub_39: "f32[1, 512]", ne: "b8[1]", sub_41: "f32[1, 512]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_4: "i64[1, 1]", ne_8: "b8[1, 1]", where_6: "i64[1, 1]", permute_133: "f32[2, 768]", div_30: "f32[1, 512, 1]", permute_137: "f32[768, 3072]", permute_141: "f32[3072, 768]", div_31: "f32[1, 512, 1]", permute_145: "f32[768, 768]", permute_157: "f32[768, 768]", permute_162: "f32[768, 768]", permute_166: "f32[768, 768]", div_33: "f32[1, 512, 1]", permute_170: "f32[768, 3072]", permute_174: "f32[3072, 768]", div_34: "f32[1, 512, 1]", permute_178: "f32[768, 768]", permute_190: "f32[768, 768]", permute_195: "f32[768, 768]", permute_199: "f32[768, 768]", div_36: "f32[1, 512, 1]", permute_203: "f32[768, 3072]", permute_207: "f32[3072, 768]", div_37: "f32[1, 512, 1]", permute_211: "f32[768, 768]", permute_223: "f32[768, 768]", permute_228: "f32[768, 768]", permute_232: "f32[768, 768]", div_39: "f32[1, 512, 1]", permute_236: "f32[768, 3072]", permute_240: "f32[3072, 768]", div_40: "f32[1, 512, 1]", permute_244: "f32[768, 768]", permute_256: "f32[768, 768]", permute_261: "f32[768, 768]", permute_265: "f32[768, 768]", div_42: "f32[1, 512, 1]", permute_269: "f32[768, 3072]", permute_273: "f32[3072, 768]", div_43: "f32[1, 512, 1]", permute_277: "f32[768, 768]", permute_289: "f32[768, 768]", permute_294: "f32[768, 768]", permute_298: "f32[768, 768]", div_45: "f32[1, 512, 1]", permute_302: "f32[768, 3072]", permute_306: "f32[3072, 768]", div_46: "f32[1, 512, 1]", permute_310: "f32[768, 768]", permute_322: "f32[768, 768]", permute_327: "f32[768, 768]", permute_331: "f32[768, 768]", div_48: "f32[1, 512, 1]", permute_335: "f32[768, 3072]", permute_339: "f32[3072, 768]", div_49: "f32[1, 512, 1]", permute_343: "f32[768, 768]", permute_355: "f32[768, 768]", permute_360: "f32[768, 768]", permute_364: "f32[768, 768]", div_51: "f32[1, 512, 1]", permute_368: "f32[768, 3072]", permute_372: "f32[3072, 768]", div_52: "f32[1, 512, 1]", permute_376: "f32[768, 768]", permute_388: "f32[768, 768]", permute_393: "f32[768, 768]", permute_397: "f32[768, 768]", div_54: "f32[1, 512, 1]", permute_401: "f32[768, 3072]", permute_405: "f32[3072, 768]", div_55: "f32[1, 512, 1]", permute_409: "f32[768, 768]", permute_421: "f32[768, 768]", permute_426: "f32[768, 768]", permute_430: "f32[768, 768]", div_57: "f32[1, 512, 1]", permute_434: "f32[768, 3072]", permute_438: "f32[3072, 768]", div_58: "f32[1, 512, 1]", permute_442: "f32[768, 768]", permute_454: "f32[768, 768]", permute_459: "f32[768, 768]", permute_463: "f32[768, 768]", div_60: "f32[1, 512, 1]", permute_467: "f32[768, 3072]", permute_471: "f32[3072, 768]", div_61: "f32[1, 512, 1]", permute_475: "f32[768, 768]", permute_487: "f32[768, 768]", permute_492: "f32[768, 768]", permute_496: "f32[768, 768]", div_63: "f32[1, 512, 1]", permute_500: "f32[768, 3072]", permute_504: "f32[3072, 768]", div_64: "f32[1, 512, 1]", permute_508: "f32[768, 768]", permute_520: "f32[768, 768]", permute_525: "f32[768, 768]", permute_529: "f32[768, 768]", div_66: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512]", tangents_3: "f32[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1876, code: start_loss = loss_fct(start_logits, start_positions)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1877, code: end_loss = loss_fct(end_logits, end_positions)
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1878, code: total_loss = (start_loss + end_loss) / 2
    div_27: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1877, code: end_loss = loss_fct(end_logits, end_positions)
    div_28: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_1);  convert_element_type_1 = None
    full_default_6: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_4, -1.0);  where_4 = None
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_28, full_default_2);  ne_6 = div_28 = None
    mul_87: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [1], True)
    mul_88: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_42: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1877, code: end_loss = loss_fct(end_logits, end_positions)
    add_101: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1876, code: start_loss = loss_fct(start_logits, start_positions)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type);  div_27 = convert_element_type = None
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_6, -1.0);  full_default_6 = where_6 = None
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_29, full_default_2);  ne_8 = div_29 = None
    mul_89: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [1], True)
    mul_90: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_43: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1876, code: start_loss = loss_fct(start_logits, start_positions)
    add_102: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_101, 2);  add_101 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_102, 2);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1859, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:1858, code: logits = self.qa_outputs(sequence_output)
    view_266: "f32[512, 2]" = torch.ops.aten.reshape.default(cat, [512, 2]);  cat = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_266, permute_133);  permute_133 = None
    permute_134: "f32[2, 512]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_1: "f32[2, 768]" = torch.ops.aten.mm.default(permute_134, view_264);  permute_134 = view_264 = None
    permute_135: "f32[768, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[2]" = torch.ops.aten.reshape.default(sum_21, [2]);  sum_21 = None
    permute_136: "f32[2, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_268: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_92: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_268, primals_196);  primals_196 = None
    mul_93: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, 768)
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True)
    mul_94: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_92, mul_85);  mul_92 = None
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True);  mul_94 = None
    mul_95: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_23);  sum_23 = None
    sub_45: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_93, sum_22);  mul_93 = sum_22 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_95);  sub_45 = mul_95 = None
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_268, mul_85);  mul_85 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_97, [0, 1]);  mul_97 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_268, [0, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_269: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_99, [512, 768]);  mul_99 = None
    mm_2: "f32[512, 3072]" = torch.ops.aten.mm.default(view_269, permute_137);  permute_137 = None
    permute_138: "f32[768, 512]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_262);  permute_138 = view_262 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[768]" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
    permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    view_271: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_102: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_103: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_102, -0.5);  mul_102 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_103);  mul_103 = None
    mul_104: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_105: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_104);  view_261 = mul_104 = None
    add_104: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_101, mul_105);  mul_101 = mul_105 = None
    mul_106: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_271, add_104);  view_271 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_272: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_106, [512, 3072]);  mul_106 = None
    mm_4: "f32[512, 768]" = torch.ops.aten.mm.default(view_272, permute_141);  permute_141 = None
    permute_142: "f32[3072, 512]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_260);  permute_142 = view_260 = None
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[3072]" = torch.ops.aten.reshape.default(sum_27, [3072]);  sum_27 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_274: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_105: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_96, view_274);  mul_96 = view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_105, primals_190);  primals_190 = None
    mul_109: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_80);  mul_108 = None
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_29);  sum_29 = None
    sub_48: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_28);  mul_109 = sum_28 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_111);  sub_48 = mul_111 = None
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_49);  div_31 = sub_49 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_105, mul_80);  mul_80 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_105, [0, 1]);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_3: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_112, mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_275: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_115, [512, 768]);  mul_115 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_275, permute_145);  permute_145 = None
    permute_146: "f32[768, 512]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_258);  permute_146 = view_258 = None
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[768]" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
    permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_277: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_278: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_277, [1, 512, 12, 64]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_149: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_149, clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_149 = clone_default = clone_default_1 = clone_default_2 = alias_default_1 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_130: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
    getitem_131: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
    getitem_132: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_155: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_285: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_155, [1, 512, 768]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_286: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_156, [1, 512, 768]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_287: "f32[512, 768]" = torch.ops.aten.reshape.default(view_286, [512, 768]);  view_286 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_287, permute_157);  permute_157 = None
    permute_158: "f32[768, 512]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_158, view_242);  permute_158 = None
    permute_159: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[768]" = torch.ops.aten.reshape.default(sum_34, [768]);  sum_34 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_289: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_106: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_112, view_289);  mul_112 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_290: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_161, [1, 512, 768]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_291: "f32[512, 768]" = torch.ops.aten.reshape.default(view_290, [512, 768]);  view_290 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_291, permute_162);  permute_162 = None
    permute_163: "f32[768, 512]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_106, view_293);  add_106 = view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_294: "f32[512, 768]" = torch.ops.aten.reshape.default(view_285, [512, 768]);  view_285 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_294, permute_166);  permute_166 = None
    permute_167: "f32[768, 512]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = view_242 = None
    permute_168: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
    view_296: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_108: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_107, view_296);  add_107 = view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_108, primals_180);  primals_180 = None
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_78);  mul_121 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_37);  mul_122 = sum_37 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_124);  sub_52 = mul_124 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_53);  div_33 = sub_53 = None
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_108, mul_78);  mul_78 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_297: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_128, [512, 768]);  mul_128 = None
    mm_14: "f32[512, 3072]" = torch.ops.aten.mm.default(view_297, permute_170);  permute_170 = None
    permute_171: "f32[768, 512]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_15: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_171, view_240);  permute_171 = view_240 = None
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[768]" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
    permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_299: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 3072]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_131: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_132: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_134: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_133);  view_239 = mul_133 = None
    add_110: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_299, add_110);  view_299 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_300: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_135, [512, 3072]);  mul_135 = None
    mm_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_300, permute_174);  permute_174 = None
    permute_175: "f32[3072, 512]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_17: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[3072]" = torch.ops.aten.reshape.default(sum_42, [3072]);  sum_42 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_302: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_302);  mul_125 = view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_111, primals_174);  primals_174 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, 768)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_137, mul_73);  mul_137 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_44);  sum_44 = None
    sub_55: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_138, sum_43);  mul_138 = sum_43 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_140);  sub_55 = mul_140 = None
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_56);  div_34 = sub_56 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_111, mul_73);  mul_73 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_6: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_303: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_144, [512, 768]);  mul_144 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_303, permute_178);  permute_178 = None
    permute_179: "f32[768, 512]" = torch.ops.aten.permute.default(view_303, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_179, view_236);  permute_179 = view_236 = None
    permute_180: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_303, [0], True);  view_303 = None
    view_304: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
    permute_181: "f32[768, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_305: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_306: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_305, [1, 512, 12, 64]);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_182: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_306, [0, 2, 1, 3]);  view_306 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_182, clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale = 0.125);  permute_182 = clone_default_3 = clone_default_4 = clone_default_5 = alias_default_3 = getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
    getitem_138: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
    getitem_139: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_188: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_313: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_188, [1, 512, 768]);  permute_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_314: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_189, [1, 512, 768]);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_315: "f32[512, 768]" = torch.ops.aten.reshape.default(view_314, [512, 768]);  view_314 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_315, permute_190);  permute_190 = None
    permute_191: "f32[768, 512]" = torch.ops.aten.permute.default(view_315, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_191, view_220);  permute_191 = None
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[768]" = torch.ops.aten.reshape.default(sum_49, [768]);  sum_49 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_317: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_112: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_141, view_317);  mul_141 = view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_318: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_194, [1, 512, 768]);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_319: "f32[512, 768]" = torch.ops.aten.reshape.default(view_318, [512, 768]);  view_318 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_319, permute_195);  permute_195 = None
    permute_196: "f32[768, 512]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
    view_320: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_321);  add_112 = view_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_322: "f32[512, 768]" = torch.ops.aten.reshape.default(view_313, [512, 768]);  view_313 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_322, permute_199);  permute_199 = None
    permute_200: "f32[768, 512]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = view_220 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_324: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_114: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_113, view_324);  add_113 = view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_150: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, primals_164);  primals_164 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_71);  mul_150 = None
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, sum_53);  sum_53 = None
    sub_59: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_52);  mul_151 = sum_52 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_153);  sub_59 = mul_153 = None
    mul_154: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_60);  div_36 = sub_60 = None
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_114, mul_71);  mul_71 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_154, mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_325: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_157, [512, 768]);  mul_157 = None
    mm_26: "f32[512, 3072]" = torch.ops.aten.mm.default(view_325, permute_203);  permute_203 = None
    permute_204: "f32[768, 512]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
    permute_206: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    view_327: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_160: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_161: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_161);  mul_161 = None
    mul_162: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_163: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_162);  view_217 = mul_162 = None
    add_116: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
    mul_164: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_327, add_116);  view_327 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_328: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_164, [512, 3072]);  mul_164 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_328, permute_207);  permute_207 = None
    permute_208: "f32[3072, 512]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_208, view_216);  permute_208 = view_216 = None
    permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[3072]" = torch.ops.aten.reshape.default(sum_57, [3072]);  sum_57 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
    view_330: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_154, view_330);  mul_154 = view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_166: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_158);  primals_158 = None
    mul_167: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, 768)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_166, mul_66);  mul_166 = None
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_59);  sum_59 = None
    sub_62: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_167, sum_58);  mul_167 = sum_58 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_169);  sub_62 = mul_169 = None
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_63);  div_37 = sub_63 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_117, mul_66);  mul_66 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_9: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_170, mul_172);  mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_331: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_173, [512, 768]);  mul_173 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_331, permute_211);  permute_211 = None
    permute_212: "f32[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_212, view_214);  permute_212 = view_214 = None
    permute_213: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
    view_333: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_334: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_333, [1, 512, 12, 64]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_215: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_215, clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale = 0.125);  permute_215 = clone_default_6 = clone_default_7 = clone_default_8 = alias_default_5 = getitem_141 = getitem_142 = getitem_143 = None
    getitem_144: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
    getitem_145: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
    getitem_146: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_221: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_341: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_221, [1, 512, 768]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_342: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_222, [1, 512, 768]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_343: "f32[512, 768]" = torch.ops.aten.reshape.default(view_342, [512, 768]);  view_342 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_343, permute_223);  permute_223 = None
    permute_224: "f32[768, 512]" = torch.ops.aten.permute.default(view_343, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_224, view_198);  permute_224 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True);  view_343 = None
    view_344: "f32[768]" = torch.ops.aten.reshape.default(sum_64, [768]);  sum_64 = None
    permute_226: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_345: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_170, view_345);  mul_170 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_346: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_227, [1, 512, 768]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_347: "f32[512, 768]" = torch.ops.aten.reshape.default(view_346, [512, 768]);  view_346 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_347, permute_228);  permute_228 = None
    permute_229: "f32[768, 512]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_118, view_349);  add_118 = view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_350: "f32[512, 768]" = torch.ops.aten.reshape.default(view_341, [512, 768]);  view_341 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_350, permute_232);  permute_232 = None
    permute_233: "f32[768, 512]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = view_198 = None
    permute_234: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
    view_352: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_120: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_119, view_352);  add_119 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_179: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_148);  primals_148 = None
    mul_180: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, 768)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_179, mul_64);  mul_179 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_68);  sum_68 = None
    sub_66: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_180, sum_67);  mul_180 = sum_67 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_182);  sub_66 = mul_182 = None
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_67);  div_39 = sub_67 = None
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_120, mul_64);  mul_64 = None
    sum_69: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_183, mul_185);  mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_353: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_186, [512, 768]);  mul_186 = None
    mm_38: "f32[512, 3072]" = torch.ops.aten.mm.default(view_353, permute_236);  permute_236 = None
    permute_237: "f32[768, 512]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_39: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_237, view_196);  permute_237 = view_196 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[768]" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
    permute_239: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_355: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 3072]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_188: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_191);  view_195 = mul_191 = None
    add_122: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_355, add_122);  view_355 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_193, [512, 3072]);  mul_193 = None
    mm_40: "f32[512, 768]" = torch.ops.aten.mm.default(view_356, permute_240);  permute_240 = None
    permute_241: "f32[3072, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_41: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_241, view_194);  permute_241 = view_194 = None
    permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[3072]" = torch.ops.aten.reshape.default(sum_72, [3072]);  sum_72 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_358: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_183, view_358);  mul_183 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, primals_142);  primals_142 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, 768)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_195, mul_59);  mul_195 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_74);  sum_74 = None
    sub_69: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_196, sum_73);  mul_196 = sum_73 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_198);  sub_69 = mul_198 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_70);  div_40 = sub_70 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_123, mul_59);  mul_59 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_12: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, mul_201);  mul_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_359: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_202, [512, 768]);  mul_202 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_359, permute_244);  permute_244 = None
    permute_245: "f32[768, 512]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_245, view_192);  permute_245 = view_192 = None
    permute_246: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
    permute_247: "f32[768, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_361: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_362: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_361, [1, 512, 12, 64]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_248: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_248, clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale = 0.125);  permute_248 = clone_default_9 = clone_default_10 = clone_default_11 = alias_default_7 = getitem_148 = getitem_149 = getitem_150 = None
    getitem_151: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
    getitem_152: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
    getitem_153: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_254: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_369: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_254, [1, 512, 768]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_370: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_255, [1, 512, 768]);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_371: "f32[512, 768]" = torch.ops.aten.reshape.default(view_370, [512, 768]);  view_370 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_371, permute_256);  permute_256 = None
    permute_257: "f32[768, 512]" = torch.ops.aten.permute.default(view_371, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_257, view_176);  permute_257 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True);  view_371 = None
    view_372: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    view_373: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_124: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_199, view_373);  mul_199 = view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_374: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_260, [1, 512, 768]);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_375: "f32[512, 768]" = torch.ops.aten.reshape.default(view_374, [512, 768]);  view_374 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_375, permute_261);  permute_261 = None
    permute_262: "f32[768, 512]" = torch.ops.aten.permute.default(view_375, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True);  view_375 = None
    view_376: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_377);  add_124 = view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_378: "f32[512, 768]" = torch.ops.aten.reshape.default(view_369, [512, 768]);  view_369 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_378, permute_265);  permute_265 = None
    permute_266: "f32[768, 512]" = torch.ops.aten.permute.default(view_378, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = view_176 = None
    permute_267: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_378, [0], True);  view_378 = None
    view_379: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
    view_380: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_125, view_380);  add_125 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_208: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_132);  primals_132 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, 768)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_208, mul_57);  mul_208 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_83);  sum_83 = None
    sub_73: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_209, sum_82);  mul_209 = sum_82 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_211);  sub_73 = mul_211 = None
    mul_212: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_74);  div_42 = sub_74 = None
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, mul_57);  mul_57 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_212, mul_214);  mul_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_215, [512, 768]);  mul_215 = None
    mm_50: "f32[512, 3072]" = torch.ops.aten.mm.default(view_381, permute_269);  permute_269 = None
    permute_270: "f32[768, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_270, view_174);  permute_270 = view_174 = None
    permute_271: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
    permute_272: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_383: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_219: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_221: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_220);  view_173 = mul_220 = None
    add_128: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_383, add_128);  view_383 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_222, [512, 3072]);  mul_222 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_384, permute_273);  permute_273 = None
    permute_274: "f32[3072, 512]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_274, view_172);  permute_274 = view_172 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[3072]" = torch.ops.aten.reshape.default(sum_87, [3072]);  sum_87 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_386: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_212, view_386);  mul_212 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_126);  primals_126 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, 768)
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_224, mul_52);  mul_224 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_89);  sum_89 = None
    sub_76: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_225, sum_88);  mul_225 = sum_88 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_227);  sub_76 = mul_227 = None
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_77);  div_43 = sub_77 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_129, mul_52);  mul_52 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_387: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_231, [512, 768]);  mul_231 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_387, permute_277);  permute_277 = None
    permute_278: "f32[768, 512]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_278, view_170);  permute_278 = view_170 = None
    permute_279: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[768]" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_389: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_390: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_389, [1, 512, 12, 64]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_281: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_281, clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale = 0.125);  permute_281 = clone_default_12 = clone_default_13 = clone_default_14 = alias_default_9 = getitem_155 = getitem_156 = getitem_157 = None
    getitem_158: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
    getitem_159: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
    getitem_160: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_287: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_397: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_287, [1, 512, 768]);  permute_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_398: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_288, [1, 512, 768]);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_399: "f32[512, 768]" = torch.ops.aten.reshape.default(view_398, [512, 768]);  view_398 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_399, permute_289);  permute_289 = None
    permute_290: "f32[768, 512]" = torch.ops.aten.permute.default(view_399, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_154);  permute_290 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
    view_400: "f32[768]" = torch.ops.aten.reshape.default(sum_94, [768]);  sum_94 = None
    permute_292: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    view_401: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_228, view_401);  mul_228 = view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_402: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_293, [1, 512, 768]);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_403: "f32[512, 768]" = torch.ops.aten.reshape.default(view_402, [512, 768]);  view_402 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_403, permute_294);  permute_294 = None
    permute_295: "f32[768, 512]" = torch.ops.aten.permute.default(view_403, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True);  view_403 = None
    view_404: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_130, view_405);  add_130 = view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.reshape.default(view_397, [512, 768]);  view_397 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_406, permute_298);  permute_298 = None
    permute_299: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = view_154 = None
    permute_300: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_408: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_131, view_408);  add_131 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_237: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_116);  primals_116 = None
    mul_238: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_50);  mul_237 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_98);  sum_98 = None
    sub_80: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_97);  mul_238 = sum_97 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_240);  sub_80 = mul_240 = None
    mul_241: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_81);  div_45 = sub_81 = None
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, mul_50);  mul_50 = None
    sum_99: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_241, mul_243);  mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_409: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_244, [512, 768]);  mul_244 = None
    mm_62: "f32[512, 3072]" = torch.ops.aten.mm.default(view_409, permute_302);  permute_302 = None
    permute_303: "f32[768, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_63: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_303, view_152);  permute_303 = view_152 = None
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
    permute_305: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_411: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_62, [1, 512, 3072]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_246: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_247: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_248: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_247, -0.5);  mul_247 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_248);  mul_248 = None
    mul_249: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_250: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_249);  view_151 = mul_249 = None
    add_134: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_246, mul_250);  mul_246 = mul_250 = None
    mul_251: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_411, add_134);  view_411 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_412: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_251, [512, 3072]);  mul_251 = None
    mm_64: "f32[512, 768]" = torch.ops.aten.mm.default(view_412, permute_306);  permute_306 = None
    permute_307: "f32[3072, 512]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_65: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_307, view_150);  permute_307 = view_150 = None
    permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
    view_413: "f32[3072]" = torch.ops.aten.reshape.default(sum_102, [3072]);  sum_102 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_308, [1, 0]);  permute_308 = None
    view_414: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_64, [1, 512, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_241, view_414);  mul_241 = view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, primals_110);  primals_110 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, 768)
    sum_103: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_253, mul_45);  mul_253 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_104);  sum_104 = None
    sub_83: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_254, sum_103);  mul_254 = sum_103 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_256);  sub_83 = mul_256 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_84);  div_46 = sub_84 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_135, mul_45);  mul_45 = None
    sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_415: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_260, [512, 768]);  mul_260 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_415, permute_310);  permute_310 = None
    permute_311: "f32[768, 512]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_311, view_148);  permute_311 = view_148 = None
    permute_312: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
    permute_313: "f32[768, 768]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
    view_417: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_418: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_417, [1, 512, 12, 64]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_314: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_314, clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale = 0.125);  permute_314 = clone_default_15 = clone_default_16 = clone_default_17 = alias_default_11 = getitem_162 = getitem_163 = getitem_164 = None
    getitem_165: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
    getitem_166: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
    getitem_167: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_320: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_425: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_320, [1, 512, 768]);  permute_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_426: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_321, [1, 512, 768]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_427: "f32[512, 768]" = torch.ops.aten.reshape.default(view_426, [512, 768]);  view_426 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_427, permute_322);  permute_322 = None
    permute_323: "f32[768, 512]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_323, view_132);  permute_323 = None
    permute_324: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
    view_428: "f32[768]" = torch.ops.aten.reshape.default(sum_109, [768]);  sum_109 = None
    permute_325: "f32[768, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    view_429: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_257, view_429);  mul_257 = view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_430: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_326, [1, 512, 768]);  permute_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_431: "f32[512, 768]" = torch.ops.aten.reshape.default(view_430, [512, 768]);  view_430 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_431, permute_327);  permute_327 = None
    permute_328: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_433);  add_136 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_434: "f32[512, 768]" = torch.ops.aten.reshape.default(view_425, [512, 768]);  view_425 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_434, permute_331);  permute_331 = None
    permute_332: "f32[768, 512]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = view_132 = None
    permute_333: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    view_436: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_138: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_137, view_436);  add_137 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_266: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, primals_100);  primals_100 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, 768)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_266, mul_43);  mul_266 = None
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, sum_113);  sum_113 = None
    sub_87: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_267, sum_112);  mul_267 = sum_112 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_269);  sub_87 = mul_269 = None
    mul_270: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_88);  div_48 = sub_88 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_138, mul_43);  mul_43 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_437: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_273, [512, 768]);  mul_273 = None
    mm_74: "f32[512, 3072]" = torch.ops.aten.mm.default(view_437, permute_335);  permute_335 = None
    permute_336: "f32[768, 512]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_336, view_130);  permute_336 = view_130 = None
    permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[768]" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
    permute_338: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
    view_439: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_74, [1, 512, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_275: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_276: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_277: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
    mul_278: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_279: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_278);  view_129 = mul_278 = None
    add_140: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
    mul_280: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_439, add_140);  view_439 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_440: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_280, [512, 3072]);  mul_280 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_440, permute_339);  permute_339 = None
    permute_340: "f32[3072, 512]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_340, view_128);  permute_340 = view_128 = None
    permute_341: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[3072]" = torch.ops.aten.reshape.default(sum_117, [3072]);  sum_117 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
    view_442: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_270, view_442);  mul_270 = view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_282: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_94);  primals_94 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, 768)
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, mul_38);  mul_282 = None
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_119);  sum_119 = None
    sub_90: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_283, sum_118);  mul_283 = sum_118 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_285);  sub_90 = mul_285 = None
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_91);  div_49 = sub_91 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_141, mul_38);  mul_38 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_443: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_289, [512, 768]);  mul_289 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_443, permute_343);  permute_343 = None
    permute_344: "f32[768, 512]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_344, view_126);  permute_344 = view_126 = None
    permute_345: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[768]" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
    view_445: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_446: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_445, [1, 512, 12, 64]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_347: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_347, clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale = 0.125);  permute_347 = clone_default_18 = clone_default_19 = clone_default_20 = alias_default_13 = getitem_169 = getitem_170 = getitem_171 = None
    getitem_172: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
    getitem_173: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
    getitem_174: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_353: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_453: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_353, [1, 512, 768]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_454: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_354, [1, 512, 768]);  permute_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_455: "f32[512, 768]" = torch.ops.aten.reshape.default(view_454, [512, 768]);  view_454 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_455, permute_355);  permute_355 = None
    permute_356: "f32[768, 512]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_356, view_110);  permute_356 = None
    permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
    view_456: "f32[768]" = torch.ops.aten.reshape.default(sum_124, [768]);  sum_124 = None
    permute_358: "f32[768, 768]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_457: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_286, view_457);  mul_286 = view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_458: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_359, [1, 512, 768]);  permute_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_459: "f32[512, 768]" = torch.ops.aten.reshape.default(view_458, [512, 768]);  view_458 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_360);  permute_360 = None
    permute_361: "f32[768, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_461);  add_142 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_462: "f32[512, 768]" = torch.ops.aten.reshape.default(view_453, [512, 768]);  view_453 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_462, permute_364);  permute_364 = None
    permute_365: "f32[768, 512]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = view_110 = None
    permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(permute_366, [1, 0]);  permute_366 = None
    view_464: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_144: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_143, view_464);  add_143 = view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_84);  primals_84 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_295, 768)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_295, mul_36);  mul_295 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_128);  sum_128 = None
    sub_94: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_296, sum_127);  mul_296 = sum_127 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_298);  sub_94 = mul_298 = None
    mul_299: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_95);  div_51 = sub_95 = None
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_144, mul_36);  mul_36 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_299, mul_301);  mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_465: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_302, [512, 768]);  mul_302 = None
    mm_86: "f32[512, 3072]" = torch.ops.aten.mm.default(view_465, permute_368);  permute_368 = None
    permute_369: "f32[768, 512]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_87: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_108);  permute_369 = view_108 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[768]" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_467: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_86, [1, 512, 3072]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_304: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_305: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_306: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_308: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_307);  view_107 = mul_307 = None
    add_146: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_467, add_146);  view_467 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_468: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_309, [512, 3072]);  mul_309 = None
    mm_88: "f32[512, 768]" = torch.ops.aten.mm.default(view_468, permute_372);  permute_372 = None
    permute_373: "f32[3072, 512]" = torch.ops.aten.permute.default(view_468, [1, 0])
    mm_89: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_106);  permute_373 = view_106 = None
    permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
    view_469: "f32[3072]" = torch.ops.aten.reshape.default(sum_132, [3072]);  sum_132 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_470: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_88, [1, 512, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_299, view_470);  mul_299 = view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_311: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, primals_78);  primals_78 = None
    mul_312: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_31);  mul_311 = None
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, sum_134);  sum_134 = None
    sub_97: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_133);  mul_312 = sum_133 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_314);  sub_97 = mul_314 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_98);  div_52 = sub_98 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_31);  mul_31 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_315, mul_317);  mul_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_471: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_318, [512, 768]);  mul_318 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_471, permute_376);  permute_376 = None
    permute_377: "f32[768, 512]" = torch.ops.aten.permute.default(view_471, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_377, view_104);  permute_377 = view_104 = None
    permute_378: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
    view_472: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_473: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_474: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_473, [1, 512, 12, 64]);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_380: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_474, [0, 2, 1, 3]);  view_474 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_380, clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale = 0.125);  permute_380 = clone_default_21 = clone_default_22 = clone_default_23 = alias_default_15 = getitem_176 = getitem_177 = getitem_178 = None
    getitem_179: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
    getitem_180: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
    getitem_181: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_386: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_481: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_386, [1, 512, 768]);  permute_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_482: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_387, [1, 512, 768]);  permute_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_483: "f32[512, 768]" = torch.ops.aten.reshape.default(view_482, [512, 768]);  view_482 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_483, permute_388);  permute_388 = None
    permute_389: "f32[768, 512]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_389, view_88);  permute_389 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
    view_484: "f32[768]" = torch.ops.aten.reshape.default(sum_139, [768]);  sum_139 = None
    permute_391: "f32[768, 768]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_485: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_315, view_485);  mul_315 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_486: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_392, [1, 512, 768]);  permute_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_487: "f32[512, 768]" = torch.ops.aten.reshape.default(view_486, [512, 768]);  view_486 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_487, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_148, view_489);  add_148 = view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_490: "f32[512, 768]" = torch.ops.aten.reshape.default(view_481, [512, 768]);  view_481 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_490, permute_397);  permute_397 = None
    permute_398: "f32[768, 512]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = view_88 = None
    permute_399: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
    view_492: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_150: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_149, view_492);  add_149 = view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_324: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, primals_68);  primals_68 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, 768)
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_324, mul_29);  mul_324 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_143);  sum_143 = None
    sub_101: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_325, sum_142);  mul_325 = sum_142 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_327);  sub_101 = mul_327 = None
    mul_328: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_102);  div_54 = sub_102 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_150, mul_29);  mul_29 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, mul_330);  mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_493: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_331, [512, 768]);  mul_331 = None
    mm_98: "f32[512, 3072]" = torch.ops.aten.mm.default(view_493, permute_401);  permute_401 = None
    permute_402: "f32[768, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_99: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_402, view_86);  permute_402 = view_86 = None
    permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[768]" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
    permute_404: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
    view_495: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_98, [1, 512, 3072]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_336);  view_85 = mul_336 = None
    add_152: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_495, add_152);  view_495 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_496: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_338, [512, 3072]);  mul_338 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_405);  permute_405 = None
    permute_406: "f32[3072, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_101: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_406, view_84);  permute_406 = view_84 = None
    permute_407: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[3072]" = torch.ops.aten.reshape.default(sum_147, [3072]);  sum_147 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_498: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_328, view_498);  mul_328 = view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_62);  primals_62 = None
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, 768)
    sum_148: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_340, mul_24);  mul_340 = None
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_149);  sum_149 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_341, sum_148);  mul_341 = sum_148 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_343);  sub_104 = mul_343 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_105);  div_55 = sub_105 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_24);  mul_24 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_27: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_344, mul_346);  mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_499: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_347, [512, 768]);  mul_347 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_499, permute_409);  permute_409 = None
    permute_410: "f32[768, 512]" = torch.ops.aten.permute.default(view_499, [1, 0])
    mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_410, view_82);  permute_410 = view_82 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0], True);  view_499 = None
    view_500: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
    permute_412: "f32[768, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_501: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_502: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_501, [1, 512, 12, 64]);  view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_413: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_413, clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale = 0.125);  permute_413 = clone_default_24 = clone_default_25 = clone_default_26 = alias_default_17 = getitem_183 = getitem_184 = getitem_185 = None
    getitem_186: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
    getitem_187: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
    getitem_188: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_419: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_509: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_419, [1, 512, 768]);  permute_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_510: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_420, [1, 512, 768]);  permute_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_511: "f32[512, 768]" = torch.ops.aten.reshape.default(view_510, [512, 768]);  view_510 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_511, permute_421);  permute_421 = None
    permute_422: "f32[768, 512]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_422, view_66);  permute_422 = None
    permute_423: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[768]" = torch.ops.aten.reshape.default(sum_154, [768]);  sum_154 = None
    permute_424: "f32[768, 768]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
    view_513: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_344, view_513);  mul_344 = view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_514: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_425, [1, 512, 768]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_515: "f32[512, 768]" = torch.ops.aten.reshape.default(view_514, [512, 768]);  view_514 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_426);  permute_426 = None
    permute_427: "f32[768, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_517);  add_154 = view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_518: "f32[512, 768]" = torch.ops.aten.reshape.default(view_509, [512, 768]);  view_509 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_518, permute_430);  permute_430 = None
    permute_431: "f32[768, 512]" = torch.ops.aten.permute.default(view_518, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = view_66 = None
    permute_432: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
    view_519: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_520: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_155, view_520);  add_155 = view_520 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_52);  primals_52 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, 768)
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_353, mul_22);  mul_353 = None
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_158);  sum_158 = None
    sub_108: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_354, sum_157);  mul_354 = sum_157 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_356);  sub_108 = mul_356 = None
    mul_357: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_109);  div_57 = sub_109 = None
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_22);  mul_22 = None
    sum_159: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_521: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_360, [512, 768]);  mul_360 = None
    mm_110: "f32[512, 3072]" = torch.ops.aten.mm.default(view_521, permute_434);  permute_434 = None
    permute_435: "f32[768, 512]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_111: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_435, view_64);  permute_435 = view_64 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[768]" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
    permute_437: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_523: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_110, [1, 512, 3072]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_363: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_364: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_366: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_365);  view_63 = mul_365 = None
    add_158: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_523, add_158);  view_523 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_524: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_367, [512, 3072]);  mul_367 = None
    mm_112: "f32[512, 768]" = torch.ops.aten.mm.default(view_524, permute_438);  permute_438 = None
    permute_439: "f32[3072, 512]" = torch.ops.aten.permute.default(view_524, [1, 0])
    mm_113: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_439, view_62);  permute_439 = view_62 = None
    permute_440: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_524, [0], True);  view_524 = None
    view_525: "f32[3072]" = torch.ops.aten.reshape.default(sum_162, [3072]);  sum_162 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_526: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_112, [1, 512, 768]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_357, view_526);  mul_357 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, primals_46);  primals_46 = None
    mul_370: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, 768)
    sum_163: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_369, mul_17);  mul_369 = None
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_164);  sum_164 = None
    sub_111: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_370, sum_163);  mul_370 = sum_163 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_372);  sub_111 = mul_372 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_112);  div_58 = sub_112 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_159, mul_17);  mul_17 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_30: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_373, mul_375);  mul_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_527: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_376, [512, 768]);  mul_376 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_527, permute_442);  permute_442 = None
    permute_443: "f32[768, 512]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_443, view_60);  permute_443 = view_60 = None
    permute_444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_527, [0], True);  view_527 = None
    view_528: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
    permute_445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_444, [1, 0]);  permute_444 = None
    view_529: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_530: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_529, [1, 512, 12, 64]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_446: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_446, clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale = 0.125);  permute_446 = clone_default_27 = clone_default_28 = clone_default_29 = alias_default_19 = getitem_190 = getitem_191 = getitem_192 = None
    getitem_193: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
    getitem_194: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
    getitem_195: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_452: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_537: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_452, [1, 512, 768]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_538: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_453, [1, 512, 768]);  permute_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_539: "f32[512, 768]" = torch.ops.aten.reshape.default(view_538, [512, 768]);  view_538 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_539, permute_454);  permute_454 = None
    permute_455: "f32[768, 512]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_455, view_44);  permute_455 = None
    permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_169: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[768]" = torch.ops.aten.reshape.default(sum_169, [768]);  sum_169 = None
    permute_457: "f32[768, 768]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
    view_541: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_160: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_373, view_541);  mul_373 = view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_542: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_458, [1, 512, 768]);  permute_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_543: "f32[512, 768]" = torch.ops.aten.reshape.default(view_542, [512, 768]);  view_542 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_543, permute_459);  permute_459 = None
    permute_460: "f32[768, 512]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_160, view_545);  add_160 = view_545 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_546: "f32[512, 768]" = torch.ops.aten.reshape.default(view_537, [512, 768]);  view_537 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_546, permute_463);  permute_463 = None
    permute_464: "f32[768, 512]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = view_44 = None
    permute_465: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
    view_547: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
    view_548: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_161, view_548);  add_161 = view_548 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_382: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, primals_36);  primals_36 = None
    mul_383: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_382, 768)
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
    mul_384: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_382, mul_15);  mul_382 = None
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
    mul_385: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_173);  sum_173 = None
    sub_115: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_383, sum_172);  mul_383 = sum_172 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_385);  sub_115 = mul_385 = None
    mul_386: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_116);  div_60 = sub_116 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, mul_15);  mul_15 = None
    sum_174: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_386, mul_388);  mul_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_549: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_389, [512, 768]);  mul_389 = None
    mm_122: "f32[512, 3072]" = torch.ops.aten.mm.default(view_549, permute_467);  permute_467 = None
    permute_468: "f32[768, 512]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_123: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_468, view_42);  permute_468 = view_42 = None
    permute_469: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_176: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[768]" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
    permute_470: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_469, [1, 0]);  permute_469 = None
    view_551: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_122, [1, 512, 3072]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_391: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_392: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_393: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
    mul_394: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_395: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_394);  view_41 = mul_394 = None
    add_164: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
    mul_396: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_551, add_164);  view_551 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_552: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_396, [512, 3072]);  mul_396 = None
    mm_124: "f32[512, 768]" = torch.ops.aten.mm.default(view_552, permute_471);  permute_471 = None
    permute_472: "f32[3072, 512]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_125: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_472, view_40);  permute_472 = view_40 = None
    permute_473: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[3072]" = torch.ops.aten.reshape.default(sum_177, [3072]);  sum_177 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
    view_554: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_124, [1, 512, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_386, view_554);  mul_386 = view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, primals_30);  primals_30 = None
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, 768)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_398, mul_10);  mul_398 = None
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_179);  sum_179 = None
    sub_118: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_399, sum_178);  mul_399 = sum_178 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_401);  sub_118 = mul_401 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_119);  div_61 = sub_119 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_165, mul_10);  mul_10 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_33: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_555: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_405, [512, 768]);  mul_405 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_555, permute_475);  permute_475 = None
    permute_476: "f32[768, 512]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_476, view_38);  permute_476 = view_38 = None
    permute_477: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
    permute_478: "f32[768, 768]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_557: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_558: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_557, [1, 512, 12, 64]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_479: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_479, clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale = 0.125);  permute_479 = clone_default_30 = clone_default_31 = clone_default_32 = alias_default_21 = getitem_197 = getitem_198 = getitem_199 = None
    getitem_200: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
    getitem_201: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
    getitem_202: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_485: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_565: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_485, [1, 512, 768]);  permute_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_566: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_486, [1, 512, 768]);  permute_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_567: "f32[512, 768]" = torch.ops.aten.reshape.default(view_566, [512, 768]);  view_566 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_567, permute_487);  permute_487 = None
    permute_488: "f32[768, 512]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_488, view_22);  permute_488 = None
    permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_184: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.reshape.default(sum_184, [768]);  sum_184 = None
    permute_490: "f32[768, 768]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    view_569: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_402, view_569);  mul_402 = view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_570: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_491, [1, 512, 768]);  permute_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_571: "f32[512, 768]" = torch.ops.aten.reshape.default(view_570, [512, 768]);  view_570 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_492);  permute_492 = None
    permute_493: "f32[768, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_573);  add_166 = view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_574: "f32[512, 768]" = torch.ops.aten.reshape.default(view_565, [512, 768]);  view_565 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_574, permute_496);  permute_496 = None
    permute_497: "f32[768, 512]" = torch.ops.aten.permute.default(view_574, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = view_22 = None
    permute_498: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True);  view_574 = None
    view_575: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_576: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_168: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_167, view_576);  add_167 = view_576 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, primals_20);  primals_20 = None
    mul_412: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_411, 768)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_411, mul_8);  mul_411 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_412, sum_187);  mul_412 = sum_187 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_414);  sub_122 = mul_414 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_168, mul_8);  mul_8 = None
    sum_189: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:465, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_415, mul_417);  mul_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_577: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_418, [512, 768]);  mul_418 = None
    mm_134: "f32[512, 3072]" = torch.ops.aten.mm.default(view_577, permute_500);  permute_500 = None
    permute_501: "f32[768, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_135: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_501, view_20);  permute_501 = view_20 = None
    permute_502: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[768]" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
    permute_503: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_579: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 3072]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_420: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_421: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_422: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_27: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_424: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_423);  view_19 = mul_423 = None
    add_170: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_579, add_170);  view_579 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_580: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_425, [512, 3072]);  mul_425 = None
    mm_136: "f32[512, 768]" = torch.ops.aten.mm.default(view_580, permute_504);  permute_504 = None
    permute_505: "f32[3072, 512]" = torch.ops.aten.permute.default(view_580, [1, 0])
    mm_137: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_505, view_18);  permute_505 = view_18 = None
    permute_506: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
    view_581: "f32[3072]" = torch.ops.aten.reshape.default(sum_192, [3072]);  sum_192 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None
    view_582: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_136, [1, 512, 768]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_171: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_415, view_582);  mul_415 = view_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, primals_14);  primals_14 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, 768)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_427, mul_3);  mul_427 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_428, sum_193);  mul_428 = sum_193 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_430);  sub_125 = mul_430 = None
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_3);  mul_3 = None
    sum_195: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:387, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_36: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_583: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_434, [512, 768]);  mul_434 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_583, permute_508);  permute_508 = None
    permute_509: "f32[768, 512]" = torch.ops.aten.permute.default(view_583, [1, 0])
    mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_509, view_16);  permute_509 = view_16 = None
    permute_510: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
    view_584: "f32[768]" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None
    permute_511: "f32[768, 768]" = torch.ops.aten.permute.default(permute_510, [1, 0]);  permute_510 = None
    view_585: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_586: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_585, [1, 512, 12, 64]);  view_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_512: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_512, clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale = 0.125);  permute_512 = clone_default_33 = clone_default_34 = clone_default_35 = alias_default_23 = getitem_204 = getitem_205 = getitem_206 = None
    getitem_207: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
    getitem_208: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
    getitem_209: "f32[1, 12, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_518: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_593: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_518, [1, 512, 768]);  permute_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_594: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_519, [1, 512, 768]);  permute_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_595: "f32[512, 768]" = torch.ops.aten.reshape.default(view_594, [512, 768]);  view_594 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_595, permute_520);  permute_520 = None
    permute_521: "f32[768, 512]" = torch.ops.aten.permute.default(view_595, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_521, view);  permute_521 = None
    permute_522: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True);  view_595 = None
    view_596: "f32[768]" = torch.ops.aten.reshape.default(sum_199, [768]);  sum_199 = None
    permute_523: "f32[768, 768]" = torch.ops.aten.permute.default(permute_522, [1, 0]);  permute_522 = None
    view_597: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_431, view_597);  mul_431 = view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_598: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_524, [1, 512, 768]);  permute_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_599: "f32[512, 768]" = torch.ops.aten.reshape.default(view_598, [512, 768]);  view_598 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_599, permute_525);  permute_525 = None
    permute_526: "f32[768, 512]" = torch.ops.aten.permute.default(view_599, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True);  view_599 = None
    view_600: "f32[768]" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_601);  add_172 = view_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_602: "f32[512, 768]" = torch.ops.aten.reshape.default(view_593, [512, 768]);  view_593 = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_602, permute_529);  permute_529 = None
    permute_530: "f32[768, 512]" = torch.ops.aten.permute.default(view_602, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = view = None
    permute_531: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
    view_603: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(permute_531, [1, 0]);  permute_531 = None
    view_604: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_174: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_173, view_604);  add_173 = view_604 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:240, code: embeddings = self.dropout(embeddings)
    convert_element_type_38: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_439: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_440: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_174, mul_439);  add_174 = mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    mul_442: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_440, primals_4);  primals_4 = None
    mul_443: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, 768)
    sum_202: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True)
    mul_444: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_442, mul_1);  mul_442 = None
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_203);  sum_203 = None
    sub_129: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_443, sum_202);  mul_443 = sum_202 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_129, mul_445);  sub_129 = mul_445 = None
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_130);  div_66 = sub_130 = None
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_440, mul_1);  mul_1 = None
    sum_204: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 1]);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_8: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_8, full_default_2, mul_446);  unsqueeze_8 = None
    full_default_12: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_12, [slice_4], where_8, True);  full_default_12 = slice_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_9: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_9, full_default_2, mul_446);  unsqueeze_9 = None
    full_default_14: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_14, [expand], where_9, True);  full_default_14 = expand = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_202, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_10: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, mul_446);  unsqueeze_10 = full_default_2 = mul_446 = None
    full_default_16: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_16, [primals_202], where_10, True);  full_default_16 = primals_202 = where_10 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_204, sum_205, permute_532, view_603, permute_528, view_600, permute_523, view_596, permute_511, view_584, sum_195, sum_196, permute_507, view_581, permute_503, view_578, sum_189, sum_190, permute_499, view_575, permute_495, view_572, permute_490, view_568, permute_478, view_556, sum_180, sum_181, permute_474, view_553, permute_470, view_550, sum_174, sum_175, permute_466, view_547, permute_462, view_544, permute_457, view_540, permute_445, view_528, sum_165, sum_166, permute_441, view_525, permute_437, view_522, sum_159, sum_160, permute_433, view_519, permute_429, view_516, permute_424, view_512, permute_412, view_500, sum_150, sum_151, permute_408, view_497, permute_404, view_494, sum_144, sum_145, permute_400, view_491, permute_396, view_488, permute_391, view_484, permute_379, view_472, sum_135, sum_136, permute_375, view_469, permute_371, view_466, sum_129, sum_130, permute_367, view_463, permute_363, view_460, permute_358, view_456, permute_346, view_444, sum_120, sum_121, permute_342, view_441, permute_338, view_438, sum_114, sum_115, permute_334, view_435, permute_330, view_432, permute_325, view_428, permute_313, view_416, sum_105, sum_106, permute_309, view_413, permute_305, view_410, sum_99, sum_100, permute_301, view_407, permute_297, view_404, permute_292, view_400, permute_280, view_388, sum_90, sum_91, permute_276, view_385, permute_272, view_382, sum_84, sum_85, permute_268, view_379, permute_264, view_376, permute_259, view_372, permute_247, view_360, sum_75, sum_76, permute_243, view_357, permute_239, view_354, sum_69, sum_70, permute_235, view_351, permute_231, view_348, permute_226, view_344, permute_214, view_332, sum_60, sum_61, permute_210, view_329, permute_206, view_326, sum_54, sum_55, permute_202, view_323, permute_198, view_320, permute_193, view_316, permute_181, view_304, sum_45, sum_46, permute_177, view_301, permute_173, view_298, sum_39, sum_40, permute_169, view_295, permute_165, view_292, permute_160, view_288, permute_148, view_276, sum_30, sum_31, permute_144, view_273, permute_140, view_270, sum_24, sum_25, permute_136, view_267, None, None, None, None, None]
    