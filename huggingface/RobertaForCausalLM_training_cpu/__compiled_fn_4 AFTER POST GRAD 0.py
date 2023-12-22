from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_14: "f32[768]", primals_20: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_46: "f32[768]", primals_52: "f32[768]", primals_62: "f32[768]", primals_68: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_94: "f32[768]", primals_100: "f32[768]", primals_110: "f32[768]", primals_116: "f32[768]", primals_126: "f32[768]", primals_132: "f32[768]", primals_142: "f32[768]", primals_148: "f32[768]", primals_158: "f32[768]", primals_164: "f32[768]", primals_174: "f32[768]", primals_180: "f32[768]", primals_190: "f32[768]", primals_196: "f32[768]", primals_200: "f32[768]", primals_206: "i64[1, 512]", expand: "i64[1, 512]", add_1: "i64[1, 512]", mul_2: "f32[1, 512, 768]", getitem_3: "b8[1, 512, 768]", view: "f32[512, 768]", getitem_149: "b8[1, 12, 512, 512]", permute_default_67: "f32[12, 512, 512]", permute_default_68: "f32[12, 64, 512]", alias_default_23: "f32[1, 12, 512, 512]", permute_default_69: "f32[12, 64, 512]", permute_default_70: "f32[12, 512, 64]", view_16: "f32[512, 768]", getitem_7: "b8[1, 512, 768]", mul_4: "f32[1, 512, 768]", view_18: "f32[512, 768]", addmm_4: "f32[512, 3072]", view_20: "f32[512, 3072]", getitem_11: "b8[1, 512, 768]", mul_9: "f32[1, 512, 768]", view_22: "f32[512, 768]", getitem_147: "b8[1, 12, 512, 512]", permute_default_61: "f32[12, 512, 512]", permute_default_62: "f32[12, 64, 512]", alias_default_21: "f32[1, 12, 512, 512]", permute_default_63: "f32[12, 64, 512]", permute_default_64: "f32[12, 512, 64]", view_38: "f32[512, 768]", getitem_17: "b8[1, 512, 768]", mul_11: "f32[1, 512, 768]", view_40: "f32[512, 768]", addmm_10: "f32[512, 3072]", view_42: "f32[512, 3072]", getitem_21: "b8[1, 512, 768]", mul_16: "f32[1, 512, 768]", view_44: "f32[512, 768]", getitem_145: "b8[1, 12, 512, 512]", permute_default_55: "f32[12, 512, 512]", permute_default_56: "f32[12, 64, 512]", alias_default_19: "f32[1, 12, 512, 512]", permute_default_57: "f32[12, 64, 512]", permute_default_58: "f32[12, 512, 64]", view_60: "f32[512, 768]", getitem_27: "b8[1, 512, 768]", mul_18: "f32[1, 512, 768]", view_62: "f32[512, 768]", addmm_16: "f32[512, 3072]", view_64: "f32[512, 3072]", getitem_31: "b8[1, 512, 768]", mul_23: "f32[1, 512, 768]", view_66: "f32[512, 768]", getitem_143: "b8[1, 12, 512, 512]", permute_default_49: "f32[12, 512, 512]", permute_default_50: "f32[12, 64, 512]", alias_default_17: "f32[1, 12, 512, 512]", permute_default_51: "f32[12, 64, 512]", permute_default_52: "f32[12, 512, 64]", view_82: "f32[512, 768]", getitem_37: "b8[1, 512, 768]", mul_25: "f32[1, 512, 768]", view_84: "f32[512, 768]", addmm_22: "f32[512, 3072]", view_86: "f32[512, 3072]", getitem_41: "b8[1, 512, 768]", mul_30: "f32[1, 512, 768]", view_88: "f32[512, 768]", getitem_141: "b8[1, 12, 512, 512]", permute_default_43: "f32[12, 512, 512]", permute_default_44: "f32[12, 64, 512]", alias_default_15: "f32[1, 12, 512, 512]", permute_default_45: "f32[12, 64, 512]", permute_default_46: "f32[12, 512, 64]", view_104: "f32[512, 768]", getitem_47: "b8[1, 512, 768]", mul_32: "f32[1, 512, 768]", view_106: "f32[512, 768]", addmm_28: "f32[512, 3072]", view_108: "f32[512, 3072]", getitem_51: "b8[1, 512, 768]", mul_37: "f32[1, 512, 768]", view_110: "f32[512, 768]", getitem_139: "b8[1, 12, 512, 512]", permute_default_37: "f32[12, 512, 512]", permute_default_38: "f32[12, 64, 512]", alias_default_13: "f32[1, 12, 512, 512]", permute_default_39: "f32[12, 64, 512]", permute_default_40: "f32[12, 512, 64]", view_126: "f32[512, 768]", getitem_57: "b8[1, 512, 768]", mul_39: "f32[1, 512, 768]", view_128: "f32[512, 768]", addmm_34: "f32[512, 3072]", view_130: "f32[512, 3072]", getitem_61: "b8[1, 512, 768]", mul_44: "f32[1, 512, 768]", view_132: "f32[512, 768]", getitem_137: "b8[1, 12, 512, 512]", permute_default_31: "f32[12, 512, 512]", permute_default_32: "f32[12, 64, 512]", alias_default_11: "f32[1, 12, 512, 512]", permute_default_33: "f32[12, 64, 512]", permute_default_34: "f32[12, 512, 64]", view_148: "f32[512, 768]", getitem_67: "b8[1, 512, 768]", mul_46: "f32[1, 512, 768]", view_150: "f32[512, 768]", addmm_40: "f32[512, 3072]", view_152: "f32[512, 3072]", getitem_71: "b8[1, 512, 768]", mul_51: "f32[1, 512, 768]", view_154: "f32[512, 768]", getitem_135: "b8[1, 12, 512, 512]", permute_default_25: "f32[12, 512, 512]", permute_default_26: "f32[12, 64, 512]", alias_default_9: "f32[1, 12, 512, 512]", permute_default_27: "f32[12, 64, 512]", permute_default_28: "f32[12, 512, 64]", view_170: "f32[512, 768]", getitem_77: "b8[1, 512, 768]", mul_53: "f32[1, 512, 768]", view_172: "f32[512, 768]", addmm_46: "f32[512, 3072]", view_174: "f32[512, 3072]", getitem_81: "b8[1, 512, 768]", mul_58: "f32[1, 512, 768]", view_176: "f32[512, 768]", getitem_133: "b8[1, 12, 512, 512]", permute_default_19: "f32[12, 512, 512]", permute_default_20: "f32[12, 64, 512]", alias_default_7: "f32[1, 12, 512, 512]", permute_default_21: "f32[12, 64, 512]", permute_default_22: "f32[12, 512, 64]", view_192: "f32[512, 768]", getitem_87: "b8[1, 512, 768]", mul_60: "f32[1, 512, 768]", view_194: "f32[512, 768]", addmm_52: "f32[512, 3072]", view_196: "f32[512, 3072]", getitem_91: "b8[1, 512, 768]", mul_65: "f32[1, 512, 768]", view_198: "f32[512, 768]", getitem_131: "b8[1, 12, 512, 512]", permute_default_13: "f32[12, 512, 512]", permute_default_14: "f32[12, 64, 512]", alias_default_5: "f32[1, 12, 512, 512]", permute_default_15: "f32[12, 64, 512]", permute_default_16: "f32[12, 512, 64]", view_214: "f32[512, 768]", getitem_97: "b8[1, 512, 768]", mul_67: "f32[1, 512, 768]", view_216: "f32[512, 768]", addmm_58: "f32[512, 3072]", view_218: "f32[512, 3072]", getitem_101: "b8[1, 512, 768]", mul_72: "f32[1, 512, 768]", view_220: "f32[512, 768]", getitem_129: "b8[1, 12, 512, 512]", permute_default_7: "f32[12, 512, 512]", permute_default_8: "f32[12, 64, 512]", alias_default_3: "f32[1, 12, 512, 512]", permute_default_9: "f32[12, 64, 512]", permute_default_10: "f32[12, 512, 64]", view_236: "f32[512, 768]", getitem_107: "b8[1, 512, 768]", mul_74: "f32[1, 512, 768]", view_238: "f32[512, 768]", addmm_64: "f32[512, 3072]", view_240: "f32[512, 3072]", getitem_111: "b8[1, 512, 768]", mul_79: "f32[1, 512, 768]", view_242: "f32[512, 768]", getitem_127: "b8[1, 12, 512, 512]", permute_default_1: "f32[12, 512, 512]", permute_default_2: "f32[12, 64, 512]", alias_default_1: "f32[1, 12, 512, 512]", permute_default_3: "f32[12, 64, 512]", permute_default_4: "f32[12, 512, 64]", view_258: "f32[512, 768]", getitem_117: "b8[1, 512, 768]", mul_81: "f32[1, 512, 768]", view_260: "f32[512, 768]", addmm_70: "f32[512, 3072]", view_262: "f32[512, 3072]", getitem_121: "b8[1, 512, 768]", mul_86: "f32[1, 512, 768]", view_264: "f32[512, 768]", addmm_72: "f32[512, 768]", mul_91: "f32[1, 512, 768]", view_266: "f32[512, 768]", sub_40: "f32[511, 50265]", convert_element_type_3: "f32[]", ne_4: "b8[511, 1]", where_2: "i64[511, 1]", permute_134: "f32[50265, 768]", div_26: "f32[1, 512, 1]", permute_138: "f32[768, 768]", div_27: "f32[1, 512, 1]", permute_142: "f32[768, 3072]", permute_146: "f32[3072, 768]", div_28: "f32[1, 512, 1]", permute_150: "f32[768, 768]", permute_162: "f32[768, 768]", permute_167: "f32[768, 768]", permute_171: "f32[768, 768]", div_30: "f32[1, 512, 1]", permute_175: "f32[768, 3072]", permute_179: "f32[3072, 768]", div_31: "f32[1, 512, 1]", permute_183: "f32[768, 768]", permute_195: "f32[768, 768]", permute_200: "f32[768, 768]", permute_204: "f32[768, 768]", div_33: "f32[1, 512, 1]", permute_208: "f32[768, 3072]", permute_212: "f32[3072, 768]", div_34: "f32[1, 512, 1]", permute_216: "f32[768, 768]", permute_228: "f32[768, 768]", permute_233: "f32[768, 768]", permute_237: "f32[768, 768]", div_36: "f32[1, 512, 1]", permute_241: "f32[768, 3072]", permute_245: "f32[3072, 768]", div_37: "f32[1, 512, 1]", permute_249: "f32[768, 768]", permute_261: "f32[768, 768]", permute_266: "f32[768, 768]", permute_270: "f32[768, 768]", div_39: "f32[1, 512, 1]", permute_274: "f32[768, 3072]", permute_278: "f32[3072, 768]", div_40: "f32[1, 512, 1]", permute_282: "f32[768, 768]", permute_294: "f32[768, 768]", permute_299: "f32[768, 768]", permute_303: "f32[768, 768]", div_42: "f32[1, 512, 1]", permute_307: "f32[768, 3072]", permute_311: "f32[3072, 768]", div_43: "f32[1, 512, 1]", permute_315: "f32[768, 768]", permute_327: "f32[768, 768]", permute_332: "f32[768, 768]", permute_336: "f32[768, 768]", div_45: "f32[1, 512, 1]", permute_340: "f32[768, 3072]", permute_344: "f32[3072, 768]", div_46: "f32[1, 512, 1]", permute_348: "f32[768, 768]", permute_360: "f32[768, 768]", permute_365: "f32[768, 768]", permute_369: "f32[768, 768]", div_48: "f32[1, 512, 1]", permute_373: "f32[768, 3072]", permute_377: "f32[3072, 768]", div_49: "f32[1, 512, 1]", permute_381: "f32[768, 768]", permute_393: "f32[768, 768]", permute_398: "f32[768, 768]", permute_402: "f32[768, 768]", div_51: "f32[1, 512, 1]", permute_406: "f32[768, 3072]", permute_410: "f32[3072, 768]", div_52: "f32[1, 512, 1]", permute_414: "f32[768, 768]", permute_426: "f32[768, 768]", permute_431: "f32[768, 768]", permute_435: "f32[768, 768]", div_54: "f32[1, 512, 1]", permute_439: "f32[768, 3072]", permute_443: "f32[3072, 768]", div_55: "f32[1, 512, 1]", permute_447: "f32[768, 768]", permute_459: "f32[768, 768]", permute_464: "f32[768, 768]", permute_468: "f32[768, 768]", div_57: "f32[1, 512, 1]", permute_472: "f32[768, 3072]", permute_476: "f32[3072, 768]", div_58: "f32[1, 512, 1]", permute_480: "f32[768, 768]", permute_492: "f32[768, 768]", permute_497: "f32[768, 768]", permute_501: "f32[768, 768]", div_60: "f32[1, 512, 1]", permute_505: "f32[768, 3072]", permute_509: "f32[3072, 768]", div_61: "f32[1, 512, 1]", permute_513: "f32[768, 768]", permute_525: "f32[768, 768]", permute_530: "f32[768, 768]", permute_534: "f32[768, 768]", div_63: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 50265]"):
    # No stacktrace found for following nodes
    convert_element_type_default_11: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
    mul_tensor_44: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.1111111111111112);  convert_element_type_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_7: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_10: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_10: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_tensor_40: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1.1111111111111112);  convert_element_type_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
    add_18: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_9: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
    mul_tensor_36: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_21: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_21);  mul_21 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_8: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
    mul_tensor_32: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_28: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_34: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_7: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_tensor_28: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.1111111111111112);  convert_element_type_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_35: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_35);  mul_35 = None
    add_42: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_6: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None
    mul_tensor_24: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_42: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_50: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_5: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_tensor_20: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.1111111111111112);  convert_element_type_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [1, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_49: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_58: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_4: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
    mul_tensor_16: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [1, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_56: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_66: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_3: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None
    mul_tensor_12: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1.1111111111111112);  convert_element_type_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [1, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_74: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_2: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_tensor_8: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [1, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_70);  mul_70 = None
    add_82: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_1: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
    mul_tensor_4: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [1, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_77: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_90: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default: "f32[1, 12, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_tensor: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [1, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_84: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1130, code: x = self.dense(features)
    view_265: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [1, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_89: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_102: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:989, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_3);  tangents_1 = convert_element_type_3 = None
    full_default_4: "f32[511, 50265]" = torch.ops.aten.full.default([511, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[511, 50265]" = torch.ops.aten.scatter.value(full_default_4, 1, where_2, -1.0);  full_default_4 = where_2 = None
    where_3: "f32[511, 1]" = torch.ops.aten.where.self(ne_4, div_25, full_default_2);  ne_4 = div_25 = None
    mul_93: "f32[511, 50265]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_13: "f32[511, 50265]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_16: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [1], True)
    mul_94: "f32[511, 50265]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_41: "f32[511, 50265]" = torch.ops.aten.sub.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    view_270: "f32[1, 511, 50265]" = torch.ops.aten.reshape.default(sub_41, [1, 511, 50265]);  sub_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:986, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    full_default_6: "f32[1, 511, 50265]" = torch.ops.aten.full.default([1, 511, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_7: "f32[1, 512, 50265]" = torch.ops.aten.full.default([1, 512, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 50265]" = torch.ops.aten.slice_scatter.default(full_default_7, view_270, 1, 0, -1);  full_default_7 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:986, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    add_105: "f32[1, 512, 50265]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1135, code: x = self.decoder(x)
    view_271: "f32[512, 50265]" = torch.ops.aten.reshape.default(add_105, [512, 50265]);  add_105 = None
    mm: "f32[512, 768]" = torch.ops.aten.mm.default(view_271, permute_134);  permute_134 = None
    permute_135: "f32[50265, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_1: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[50265]" = torch.ops.aten.reshape.default(sum_17, [50265]);  sum_17 = None
    permute_137: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_273: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm, [1, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1132, code: x = self.layer_norm(x)
    mul_96: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_200);  primals_200 = None
    mul_97: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, 768)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True)
    mul_98: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_96, mul_91);  mul_96 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True);  mul_98 = None
    mul_99: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_91, sum_19);  sum_19 = None
    sub_43: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_97, sum_18);  mul_97 = sum_18 = None
    sub_44: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_99);  sub_43 = mul_99 = None
    mul_100: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_44);  div_26 = sub_44 = None
    mul_101: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_91);  mul_91 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 1]);  mul_101 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_103: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_104: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
    mul_105: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_104, -0.5);  mul_104 = None
    exp_14: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_105);  mul_105 = None
    mul_106: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_107: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_106);  view_265 = mul_106 = None
    add_107: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_103, mul_107);  mul_103 = mul_107 = None
    mul_108: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_100, add_107);  mul_100 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:1130, code: x = self.dense(features)
    view_274: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_108, [512, 768]);  mul_108 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_274, permute_138);  permute_138 = None
    permute_139: "f32[768, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_22, [768]);  sum_22 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_276: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_110: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, primals_196);  primals_196 = None
    mul_111: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, 768)
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True)
    mul_112: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_110, mul_86);  mul_110 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [2], True);  mul_112 = None
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_24);  sum_24 = None
    sub_46: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_111, sum_23);  mul_111 = sum_23 = None
    sub_47: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_46, mul_113);  sub_46 = mul_113 = None
    mul_114: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_47);  div_27 = sub_47 = None
    mul_115: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_276, mul_86);  mul_86 = None
    sum_25: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1]);  mul_115 = None
    sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_276, [0, 1]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_116: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_117: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_114, mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_117, [512, 768]);  mul_117 = None
    mm_4: "f32[512, 3072]" = torch.ops.aten.mm.default(view_277, permute_142);  permute_142 = None
    permute_143: "f32[768, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[768]" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_279: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_119: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_120: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_121: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_120, -0.5);  mul_120 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_121);  mul_121 = None
    mul_122: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_123: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_122);  view_261 = mul_122 = None
    add_109: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_119, mul_123);  mul_119 = mul_123 = None
    mul_124: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_279, add_109);  view_279 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_124, [512, 3072]);  mul_124 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_280, permute_146);  permute_146 = None
    permute_147: "f32[3072, 512]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_28: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[3072]" = torch.ops.aten.reshape.default(sum_28, [3072]);  sum_28 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_282: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_110: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_114, view_282);  mul_114 = view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_126: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_190);  primals_190 = None
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_126, 768)
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [2], True)
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_126, mul_81);  mul_126 = None
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [2], True);  mul_128 = None
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_30);  sum_30 = None
    sub_49: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_127, sum_29);  mul_127 = sum_29 = None
    sub_50: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_49, mul_129);  sub_49 = mul_129 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_50);  div_28 = sub_50 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_81);  mul_81 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1]);  mul_131 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_132);  mul_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_283: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_133, [512, 768]);  mul_133 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_283, permute_150);  permute_150 = None
    permute_151: "f32[768, 512]" = torch.ops.aten.permute.default(view_283, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
    permute_152: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
    view_284: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_285: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_286: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_285, [1, 512, 12, 64]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_154: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
    
    # No stacktrace found for following nodes
    view_default_6: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_154, [12, 512, 64]);  permute_154 = None
    bmm_default_2: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_1, view_default_6);  permute_default_1 = None
    view_default_7: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_2, [1, 12, 512, 64]);  bmm_default_2 = None
    bmm_default_3: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
    view_default_8: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_3, [1, 12, 512, 512]);  bmm_default_3 = None
    mul_tensor_1: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor);  view_default_8 = mul_tensor = None
    mul_tensor_2: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, alias_default_1);  mul_tensor_1 = None
    sum_dim_int_list_1: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    mul_tensor_3: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_1, sum_dim_int_list_1);  alias_default_1 = sum_dim_int_list_1 = None
    sub_tensor_1: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
    view_default_9: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_1, [12, 512, 512]);  sub_tensor_1 = None
    bmm_default_4: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_3, view_default_9);  permute_default_3 = None
    view_default_10: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_4, [1, 12, 64, 512]);  bmm_default_4 = None
    mul_scalar_2: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_10, 0.3535533905932738);  view_default_10 = None
    permute_default_5: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    bmm_default_5: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_9, permute_default_4);  view_default_9 = permute_default_4 = None
    view_default_11: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_5, [1, 12, 512, 64]);  bmm_default_5 = None
    mul_scalar_3: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_160: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_3, [0, 2, 1, 3]);  mul_scalar_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_293: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_15, [1, 512, 768]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_294: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_16, [1, 512, 768]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_295: "f32[512, 768]" = torch.ops.aten.reshape.default(view_294, [512, 768]);  view_294 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_295, permute_162);  permute_162 = None
    permute_163: "f32[768, 512]" = torch.ops.aten.permute.default(view_295, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True);  view_295 = None
    view_296: "f32[768]" = torch.ops.aten.reshape.default(sum_35, [768]);  sum_35 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_297: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_130, view_297);  mul_130 = view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_5, [0, 2, 1, 3]);  permute_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_298: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_166, [1, 512, 768]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_299: "f32[512, 768]" = torch.ops.aten.reshape.default(view_298, [512, 768]);  view_298 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_299, permute_167);  permute_167 = None
    permute_168: "f32[768, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_168, view_242);  permute_168 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[768]" = torch.ops.aten.reshape.default(sum_36, [768]);  sum_36 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_301: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_112: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_111, view_301);  add_111 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_302: "f32[512, 768]" = torch.ops.aten.reshape.default(view_293, [512, 768]);  view_293 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_302, permute_171);  permute_171 = None
    permute_172: "f32[768, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_37, [768]);  sum_37 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_304: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_113: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_112, view_304);  add_112 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_180);  primals_180 = None
    mul_140: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_139, 768)
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True)
    mul_141: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_139, mul_79);  mul_139 = None
    sum_39: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_141, [2], True);  mul_141 = None
    mul_142: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_79, sum_39);  sum_39 = None
    sub_53: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_140, sum_38);  mul_140 = sum_38 = None
    sub_54: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_142);  sub_53 = mul_142 = None
    mul_143: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_54);  div_30 = sub_54 = None
    mul_144: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_79);  mul_79 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1]);  mul_144 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_145: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_146: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_145);  mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_146, [512, 768]);  mul_146 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_305, permute_175);  permute_175 = None
    permute_176: "f32[768, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[768]" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_307: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_148: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_90, 0.5);  add_90 = None
    mul_149: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_150: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_149, -0.5);  mul_149 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_150);  mul_150 = None
    mul_151: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_152: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_151);  view_239 = mul_151 = None
    add_115: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_148, mul_152);  mul_148 = mul_152 = None
    mul_153: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_307, add_115);  view_307 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_153, [512, 3072]);  mul_153 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_308, permute_179);  permute_179 = None
    permute_180: "f32[3072, 512]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[3072]" = torch.ops.aten.reshape.default(sum_43, [3072]);  sum_43 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_310: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_143, view_310);  mul_143 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_155: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_174);  primals_174 = None
    mul_156: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, 768)
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_74);  mul_155 = None
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_74, sum_45);  sum_45 = None
    sub_56: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_156, sum_44);  mul_156 = sum_44 = None
    sub_57: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_158);  sub_56 = mul_158 = None
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_57);  div_31 = sub_57 = None
    mul_160: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_74);  mul_74 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_162: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_159, mul_161);  mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_311: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_162, [512, 768]);  mul_162 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_311, permute_183);  permute_183 = None
    permute_184: "f32[768, 512]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_313: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_314: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_313, [1, 512, 12, 64]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_187: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_314, [0, 2, 1, 3]);  view_314 = None
    
    # No stacktrace found for following nodes
    view_default_18: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_187, [12, 512, 64]);  permute_187 = None
    bmm_default_8: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_7, view_default_18);  permute_default_7 = None
    view_default_19: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_8, [1, 12, 512, 64]);  bmm_default_8 = None
    bmm_default_9: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_18, permute_default_8);  view_default_18 = permute_default_8 = None
    view_default_20: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_9, [1, 12, 512, 512]);  bmm_default_9 = None
    mul_tensor_5: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_20, mul_tensor_4);  view_default_20 = mul_tensor_4 = None
    mul_tensor_6: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_5, alias_default_3);  mul_tensor_5 = None
    sum_dim_int_list_3: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [-1], True)
    mul_tensor_7: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_3, sum_dim_int_list_3);  alias_default_3 = sum_dim_int_list_3 = None
    sub_tensor_3: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
    view_default_21: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_3, [12, 512, 512]);  sub_tensor_3 = None
    bmm_default_10: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_9, view_default_21);  permute_default_9 = None
    view_default_22: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_10, [1, 12, 64, 512]);  bmm_default_10 = None
    mul_scalar_6: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_22, 0.3535533905932738);  view_default_22 = None
    permute_default_11: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_6, [0, 1, 3, 2]);  mul_scalar_6 = None
    bmm_default_11: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_21, permute_default_10);  view_default_21 = permute_default_10 = None
    view_default_23: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_11, [1, 12, 512, 64]);  bmm_default_11 = None
    mul_scalar_7: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_23, 0.3535533905932738);  view_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_193: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_7, [0, 2, 1, 3]);  mul_scalar_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_321: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_20, [1, 512, 768]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_19, [0, 2, 1, 3]);  view_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_322: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_21, [1, 512, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_323: "f32[512, 768]" = torch.ops.aten.reshape.default(view_322, [512, 768]);  view_322 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_323, permute_195);  permute_195 = None
    permute_196: "f32[768, 512]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_325: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_159, view_325);  mul_159 = view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_326: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_199, [1, 512, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_327: "f32[512, 768]" = torch.ops.aten.reshape.default(view_326, [512, 768]);  view_326 = None
    mm_24: "f32[512, 768]" = torch.ops.aten.mm.default(view_327, permute_200);  permute_200 = None
    permute_201: "f32[768, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_201, view_220);  permute_201 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[768]" = torch.ops.aten.reshape.default(sum_51, [768]);  sum_51 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_329: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_118: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_117, view_329);  add_117 = view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_330: "f32[512, 768]" = torch.ops.aten.reshape.default(view_321, [512, 768]);  view_321 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_330, permute_204);  permute_204 = None
    permute_205: "f32[768, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_52, [768]);  sum_52 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_332: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_118, view_332);  add_118 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_168: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_164);  primals_164 = None
    mul_169: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, 768)
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True)
    mul_170: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_168, mul_72);  mul_168 = None
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True);  mul_170 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_72, sum_54);  sum_54 = None
    sub_60: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_169, sum_53);  mul_169 = sum_53 = None
    sub_61: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_171);  sub_60 = mul_171 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_61);  div_33 = sub_61 = None
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_72);  mul_72 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1]);  mul_173 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_174: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_172, mul_174);  mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_175, [512, 768]);  mul_175 = None
    mm_28: "f32[512, 3072]" = torch.ops.aten.mm.default(view_333, permute_208);  permute_208 = None
    permute_209: "f32[768, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[768]" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_335: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_177: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
    mul_178: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_179: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_178, -0.5);  mul_178 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_179);  mul_179 = None
    mul_180: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_181: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_180);  view_217 = mul_180 = None
    add_121: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_177, mul_181);  mul_177 = mul_181 = None
    mul_182: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_335, add_121);  view_335 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_336: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_182, [512, 3072]);  mul_182 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_336, permute_212);  permute_212 = None
    permute_213: "f32[3072, 512]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_58: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
    view_337: "f32[3072]" = torch.ops.aten.reshape.default(sum_58, [3072]);  sum_58 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_338: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_172, view_338);  mul_172 = view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_184: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_158);  primals_158 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, 768)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_184, mul_67);  mul_184 = None
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_67, sum_60);  sum_60 = None
    sub_63: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_185, sum_59);  mul_185 = sum_59 = None
    sub_64: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_63, mul_187);  sub_63 = mul_187 = None
    mul_188: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_64);  div_34 = sub_64 = None
    mul_189: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_67);  mul_67 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
    sum_62: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_190: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_191: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_190);  mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_191, [512, 768]);  mul_191 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_339, permute_216);  permute_216 = None
    permute_217: "f32[768, 512]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_342: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_341, [1, 512, 12, 64]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_220: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # No stacktrace found for following nodes
    view_default_30: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_220, [12, 512, 64]);  permute_220 = None
    bmm_default_14: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_13, view_default_30);  permute_default_13 = None
    view_default_31: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_14, [1, 12, 512, 64]);  bmm_default_14 = None
    bmm_default_15: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_30, permute_default_14);  view_default_30 = permute_default_14 = None
    view_default_32: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_15, [1, 12, 512, 512]);  bmm_default_15 = None
    mul_tensor_9: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_32, mul_tensor_8);  view_default_32 = mul_tensor_8 = None
    mul_tensor_10: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_9, alias_default_5);  mul_tensor_9 = None
    sum_dim_int_list_5: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [-1], True)
    mul_tensor_11: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_5, sum_dim_int_list_5);  alias_default_5 = sum_dim_int_list_5 = None
    sub_tensor_5: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
    view_default_33: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_5, [12, 512, 512]);  sub_tensor_5 = None
    bmm_default_16: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_15, view_default_33);  permute_default_15 = None
    view_default_34: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_16, [1, 12, 64, 512]);  bmm_default_16 = None
    mul_scalar_10: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_34, 0.3535533905932738);  view_default_34 = None
    permute_default_17: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_10, [0, 1, 3, 2]);  mul_scalar_10 = None
    bmm_default_17: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_33, permute_default_16);  view_default_33 = permute_default_16 = None
    view_default_35: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_17, [1, 12, 512, 64]);  bmm_default_17 = None
    mul_scalar_11: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_35, 0.3535533905932738);  view_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_11, [0, 2, 1, 3]);  mul_scalar_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_349: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_25, [1, 512, 768]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_31, [0, 2, 1, 3]);  view_default_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_350: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_26, [1, 512, 768]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_351: "f32[512, 768]" = torch.ops.aten.reshape.default(view_350, [512, 768]);  view_350 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_351, permute_228);  permute_228 = None
    permute_229: "f32[768, 512]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.reshape.default(sum_65, [768]);  sum_65 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_188, view_353);  mul_188 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_232: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_17, [0, 2, 1, 3]);  permute_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_354: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_232, [1, 512, 768]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_355: "f32[512, 768]" = torch.ops.aten.reshape.default(view_354, [512, 768]);  view_354 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_355, permute_233);  permute_233 = None
    permute_234: "f32[768, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_198);  permute_234 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.reshape.default(sum_66, [768]);  sum_66 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_357: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_124: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_123, view_357);  add_123 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_358: "f32[512, 768]" = torch.ops.aten.reshape.default(view_349, [512, 768]);  view_349 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_358, permute_237);  permute_237 = None
    permute_238: "f32[768, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[768]" = torch.ops.aten.reshape.default(sum_67, [768]);  sum_67 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_360: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_124, view_360);  add_124 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_148);  primals_148 = None
    mul_198: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, 768)
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_197, mul_65);  mul_197 = None
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_69);  sum_69 = None
    sub_67: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_198, sum_68);  mul_198 = sum_68 = None
    sub_68: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_200);  sub_67 = mul_200 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_68);  div_36 = sub_68 = None
    mul_202: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_65);  mul_65 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_203: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_204: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_201, mul_203);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_204, [512, 768]);  mul_204 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_361, permute_241);  permute_241 = None
    permute_242: "f32[768, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[768]" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_363: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_206: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
    mul_207: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_208: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
    mul_209: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_210: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_209);  view_195 = mul_209 = None
    add_127: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
    mul_211: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_363, add_127);  view_363 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_211, [512, 3072]);  mul_211 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_364, permute_245);  permute_245 = None
    permute_246: "f32[3072, 512]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[3072]" = torch.ops.aten.reshape.default(sum_73, [3072]);  sum_73 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_128: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_201, view_366);  mul_201 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_213: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_142);  primals_142 = None
    mul_214: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_60);  mul_213 = None
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_75);  sum_75 = None
    sub_70: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_74);  mul_214 = sum_74 = None
    sub_71: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_70, mul_216);  sub_70 = mul_216 = None
    mul_217: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_71);  div_37 = sub_71 = None
    mul_218: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_60);  mul_60 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_219);  mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_367: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_220, [512, 768]);  mul_220 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_367, permute_249);  permute_249 = None
    permute_250: "f32[768, 512]" = torch.ops.aten.permute.default(view_367, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
    view_368: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_369: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_370: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_369, [1, 512, 12, 64]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_253: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # No stacktrace found for following nodes
    view_default_42: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_253, [12, 512, 64]);  permute_253 = None
    bmm_default_20: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_19, view_default_42);  permute_default_19 = None
    view_default_43: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_20, [1, 12, 512, 64]);  bmm_default_20 = None
    bmm_default_21: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_42, permute_default_20);  view_default_42 = permute_default_20 = None
    view_default_44: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_21, [1, 12, 512, 512]);  bmm_default_21 = None
    mul_tensor_13: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_44, mul_tensor_12);  view_default_44 = mul_tensor_12 = None
    mul_tensor_14: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_13, alias_default_7);  mul_tensor_13 = None
    sum_dim_int_list_7: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [-1], True)
    mul_tensor_15: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_7, sum_dim_int_list_7);  alias_default_7 = sum_dim_int_list_7 = None
    sub_tensor_7: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
    view_default_45: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_7, [12, 512, 512]);  sub_tensor_7 = None
    bmm_default_22: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_21, view_default_45);  permute_default_21 = None
    view_default_46: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_22, [1, 12, 64, 512]);  bmm_default_22 = None
    mul_scalar_14: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_46, 0.3535533905932738);  view_default_46 = None
    permute_default_23: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_14, [0, 1, 3, 2]);  mul_scalar_14 = None
    bmm_default_23: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_45, permute_default_22);  view_default_45 = permute_default_22 = None
    view_default_47: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_23, [1, 12, 512, 64]);  bmm_default_23 = None
    mul_scalar_15: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_47, 0.3535533905932738);  view_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_259: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_15, [0, 2, 1, 3]);  mul_scalar_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_377: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_30, [1, 512, 768]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_43, [0, 2, 1, 3]);  view_default_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_31, [1, 512, 768]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_379: "f32[512, 768]" = torch.ops.aten.reshape.default(view_378, [512, 768]);  view_378 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_379, permute_261);  permute_261 = None
    permute_262: "f32[768, 512]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.reshape.default(sum_80, [768]);  sum_80 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_381: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_381);  mul_217 = view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_265: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_23, [0, 2, 1, 3]);  permute_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_382: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_265, [1, 512, 768]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_383: "f32[512, 768]" = torch.ops.aten.reshape.default(view_382, [512, 768]);  view_382 = None
    mm_48: "f32[512, 768]" = torch.ops.aten.mm.default(view_383, permute_266);  permute_266 = None
    permute_267: "f32[768, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_176);  permute_267 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[768]" = torch.ops.aten.reshape.default(sum_81, [768]);  sum_81 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_385: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_130: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_385);  add_129 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_386: "f32[512, 768]" = torch.ops.aten.reshape.default(view_377, [512, 768]);  view_377 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_386, permute_270);  permute_270 = None
    permute_271: "f32[768, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[768]" = torch.ops.aten.reshape.default(sum_82, [768]);  sum_82 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_388: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_131: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_130, view_388);  add_130 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_226: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_132);  primals_132 = None
    mul_227: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, 768)
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True)
    mul_228: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_226, mul_58);  mul_226 = None
    sum_84: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_58, sum_84);  sum_84 = None
    sub_74: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_227, sum_83);  mul_227 = sum_83 = None
    sub_75: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_229);  sub_74 = mul_229 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_39, sub_75);  div_39 = sub_75 = None
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_58);  mul_58 = None
    sum_85: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_231, [0, 1]);  mul_231 = None
    sum_86: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_232: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_233: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_230, mul_232);  mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_233, [512, 768]);  mul_233 = None
    mm_52: "f32[512, 3072]" = torch.ops.aten.mm.default(view_389, permute_274);  permute_274 = None
    permute_275: "f32[768, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[768]" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
    permute_277: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_391: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_235: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_66, 0.5);  add_66 = None
    mul_236: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_236, -0.5);  mul_236 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_237);  mul_237 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_238);  view_173 = mul_238 = None
    add_133: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_235, mul_239);  mul_235 = mul_239 = None
    mul_240: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_391, add_133);  view_391 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_240, [512, 3072]);  mul_240 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_392, permute_278);  permute_278 = None
    permute_279: "f32[3072, 512]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[3072]" = torch.ops.aten.reshape.default(sum_88, [3072]);  sum_88 = None
    permute_281: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_394: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_54, [1, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_134: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_230, view_394);  mul_230 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_242: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_126);  primals_126 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_53);  mul_242 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_53, sum_90);  sum_90 = None
    sub_77: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_89);  mul_243 = sum_89 = None
    sub_78: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_245);  sub_77 = mul_245 = None
    mul_246: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_78);  div_40 = sub_78 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_53);  mul_53 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
    sum_92: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_246, mul_248);  mul_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_395: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_249, [512, 768]);  mul_249 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_395, permute_282);  permute_282 = None
    permute_283: "f32[768, 512]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_397: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_56, [1, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_398: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_397, [1, 512, 12, 64]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    
    # No stacktrace found for following nodes
    view_default_54: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_286, [12, 512, 64]);  permute_286 = None
    bmm_default_26: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_25, view_default_54);  permute_default_25 = None
    view_default_55: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_26, [1, 12, 512, 64]);  bmm_default_26 = None
    bmm_default_27: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_54, permute_default_26);  view_default_54 = permute_default_26 = None
    view_default_56: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_27, [1, 12, 512, 512]);  bmm_default_27 = None
    mul_tensor_17: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_56, mul_tensor_16);  view_default_56 = mul_tensor_16 = None
    mul_tensor_18: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_17, alias_default_9);  mul_tensor_17 = None
    sum_dim_int_list_9: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [-1], True)
    mul_tensor_19: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_9, sum_dim_int_list_9);  alias_default_9 = sum_dim_int_list_9 = None
    sub_tensor_9: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
    view_default_57: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_9, [12, 512, 512]);  sub_tensor_9 = None
    bmm_default_28: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_27, view_default_57);  permute_default_27 = None
    view_default_58: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_28, [1, 12, 64, 512]);  bmm_default_28 = None
    mul_scalar_18: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_58, 0.3535533905932738);  view_default_58 = None
    permute_default_29: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_18, [0, 1, 3, 2]);  mul_scalar_18 = None
    bmm_default_29: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_57, permute_default_28);  view_default_57 = permute_default_28 = None
    view_default_59: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_29, [1, 12, 512, 64]);  bmm_default_29 = None
    mul_scalar_19: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_59, 0.3535533905932738);  view_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_19, [0, 2, 1, 3]);  mul_scalar_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_405: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_35, [1, 512, 768]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_55, [0, 2, 1, 3]);  view_default_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_406: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_36, [1, 512, 768]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_407: "f32[512, 768]" = torch.ops.aten.reshape.default(view_406, [512, 768]);  view_406 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_407, permute_294);  permute_294 = None
    permute_295: "f32[768, 512]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[768]" = torch.ops.aten.reshape.default(sum_95, [768]);  sum_95 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_409: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_246, view_409);  mul_246 = view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_29, [0, 2, 1, 3]);  permute_default_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_410: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_298, [1, 512, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_411: "f32[512, 768]" = torch.ops.aten.reshape.default(view_410, [512, 768]);  view_410 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_411, permute_299);  permute_299 = None
    permute_300: "f32[768, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_154);  permute_300 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_413: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_413);  add_135 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_414: "f32[512, 768]" = torch.ops.aten.reshape.default(view_405, [512, 768]);  view_405 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_414, permute_303);  permute_303 = None
    permute_304: "f32[768, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.reshape.default(sum_97, [768]);  sum_97 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_416: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_62, [1, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_136, view_416);  add_136 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_116);  primals_116 = None
    mul_256: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_51);  mul_255 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_51, sum_99);  sum_99 = None
    sub_81: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_98);  mul_256 = sum_98 = None
    sub_82: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_81, mul_258);  sub_81 = mul_258 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_82);  div_42 = sub_82 = None
    mul_260: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_51);  mul_51 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_261: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_262: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_259, mul_261);  mul_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_262, [512, 768]);  mul_262 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_417, permute_307);  permute_307 = None
    permute_308: "f32[768, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_419: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_265, -0.5);  mul_265 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_266);  mul_266 = None
    mul_267: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_268: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_267);  view_151 = mul_267 = None
    add_139: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_264, mul_268);  mul_264 = mul_268 = None
    mul_269: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_419, add_139);  view_419 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_420: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_269, [512, 3072]);  mul_269 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_420, permute_311);  permute_311 = None
    permute_312: "f32[3072, 512]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[3072]" = torch.ops.aten.reshape.default(sum_103, [3072]);  sum_103 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_422: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_140: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_259, view_422);  mul_259 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_110);  primals_110 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_46);  mul_271 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_46, sum_105);  sum_105 = None
    sub_84: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_104);  mul_272 = sum_104 = None
    sub_85: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_84, mul_274);  sub_84 = mul_274 = None
    mul_275: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_85);  div_43 = sub_85 = None
    mul_276: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_46);  mul_46 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_277);  mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_423: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_278, [512, 768]);  mul_278 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_423, permute_315);  permute_315 = None
    permute_316: "f32[768, 512]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_423, [0], True);  view_423 = None
    view_424: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_425: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_426: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_425, [1, 512, 12, 64]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # No stacktrace found for following nodes
    view_default_66: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_319, [12, 512, 64]);  permute_319 = None
    bmm_default_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_31, view_default_66);  permute_default_31 = None
    view_default_67: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_32, [1, 12, 512, 64]);  bmm_default_32 = None
    bmm_default_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_66, permute_default_32);  view_default_66 = permute_default_32 = None
    view_default_68: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_33, [1, 12, 512, 512]);  bmm_default_33 = None
    mul_tensor_21: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_68, mul_tensor_20);  view_default_68 = mul_tensor_20 = None
    mul_tensor_22: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_21, alias_default_11);  mul_tensor_21 = None
    sum_dim_int_list_11: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [-1], True)
    mul_tensor_23: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_11, sum_dim_int_list_11);  alias_default_11 = sum_dim_int_list_11 = None
    sub_tensor_11: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
    view_default_69: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_11, [12, 512, 512]);  sub_tensor_11 = None
    bmm_default_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_33, view_default_69);  permute_default_33 = None
    view_default_70: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_34, [1, 12, 64, 512]);  bmm_default_34 = None
    mul_scalar_22: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_70, 0.3535533905932738);  view_default_70 = None
    permute_default_35: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_22, [0, 1, 3, 2]);  mul_scalar_22 = None
    bmm_default_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_69, permute_default_34);  view_default_69 = permute_default_34 = None
    view_default_71: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_35, [1, 12, 512, 64]);  bmm_default_35 = None
    mul_scalar_23: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_71, 0.3535533905932738);  view_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_23, [0, 2, 1, 3]);  mul_scalar_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_433: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_40, [1, 512, 768]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_67, [0, 2, 1, 3]);  view_default_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_434: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_41, [1, 512, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_435: "f32[512, 768]" = torch.ops.aten.reshape.default(view_434, [512, 768]);  view_434 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_435, permute_327);  permute_327 = None
    permute_328: "f32[768, 512]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.reshape.default(sum_110, [768]);  sum_110 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_437: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_70, [1, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_141: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_275, view_437);  mul_275 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_438: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_331, [1, 512, 768]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_439: "f32[512, 768]" = torch.ops.aten.reshape.default(view_438, [512, 768]);  view_438 = None
    mm_72: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_332);  permute_332 = None
    permute_333: "f32[768, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_132);  permute_333 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.reshape.default(sum_111, [768]);  sum_111 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_72, [1, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_141, view_441);  add_141 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_442: "f32[512, 768]" = torch.ops.aten.reshape.default(view_433, [512, 768]);  view_433 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_442, permute_336);  permute_336 = None
    permute_337: "f32[768, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.reshape.default(sum_112, [768]);  sum_112 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_444: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_142, view_444);  add_142 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_284: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_100);  primals_100 = None
    mul_285: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_284, 768)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True)
    mul_286: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_284, mul_44);  mul_284 = None
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True);  mul_286 = None
    mul_287: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_44, sum_114);  sum_114 = None
    sub_88: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_285, sum_113);  mul_285 = sum_113 = None
    sub_89: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_287);  sub_88 = mul_287 = None
    mul_288: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_45, sub_89);  div_45 = sub_89 = None
    mul_289: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_44);  mul_44 = None
    sum_115: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 1]);  mul_289 = None
    sum_116: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_290: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_288, mul_290);  mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_291, [512, 768]);  mul_291 = None
    mm_76: "f32[512, 3072]" = torch.ops.aten.mm.default(view_445, permute_340);  permute_340 = None
    permute_341: "f32[768, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
    permute_343: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_447: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_76, [1, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_293: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
    mul_294: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_295: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
    mul_296: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_297: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_296);  view_129 = mul_296 = None
    add_145: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
    mul_298: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_447, add_145);  view_447 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_298, [512, 3072]);  mul_298 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_448, permute_344);  permute_344 = None
    permute_345: "f32[3072, 512]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_118: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[3072]" = torch.ops.aten.reshape.default(sum_118, [3072]);  sum_118 = None
    permute_347: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_450: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_78, [1, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_288, view_450);  mul_288 = view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_300: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_94);  primals_94 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, 768)
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_39);  mul_300 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_39, sum_120);  sum_120 = None
    sub_91: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_301, sum_119);  mul_301 = sum_119 = None
    sub_92: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_91, mul_303);  sub_91 = mul_303 = None
    mul_304: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_92);  div_46 = sub_92 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_39);  mul_39 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_122: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_304, mul_306);  mul_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_451: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_307, [512, 768]);  mul_307 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_348);  permute_348 = None
    permute_349: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_80, [1, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_454: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_453, [1, 512, 12, 64]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    
    # No stacktrace found for following nodes
    view_default_78: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_352, [12, 512, 64]);  permute_352 = None
    bmm_default_38: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_37, view_default_78);  permute_default_37 = None
    view_default_79: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_38, [1, 12, 512, 64]);  bmm_default_38 = None
    bmm_default_39: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_78, permute_default_38);  view_default_78 = permute_default_38 = None
    view_default_80: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_39, [1, 12, 512, 512]);  bmm_default_39 = None
    mul_tensor_25: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_80, mul_tensor_24);  view_default_80 = mul_tensor_24 = None
    mul_tensor_26: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_25, alias_default_13);  mul_tensor_25 = None
    sum_dim_int_list_13: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [-1], True)
    mul_tensor_27: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_13, sum_dim_int_list_13);  alias_default_13 = sum_dim_int_list_13 = None
    sub_tensor_13: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
    view_default_81: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_13, [12, 512, 512]);  sub_tensor_13 = None
    bmm_default_40: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_39, view_default_81);  permute_default_39 = None
    view_default_82: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_40, [1, 12, 64, 512]);  bmm_default_40 = None
    mul_scalar_26: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_82, 0.3535533905932738);  view_default_82 = None
    permute_default_41: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_26, [0, 1, 3, 2]);  mul_scalar_26 = None
    bmm_default_41: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_81, permute_default_40);  view_default_81 = permute_default_40 = None
    view_default_83: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_41, [1, 12, 512, 64]);  bmm_default_41 = None
    mul_scalar_27: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_83, 0.3535533905932738);  view_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_27, [0, 2, 1, 3]);  mul_scalar_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_45, [1, 512, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_79, [0, 2, 1, 3]);  view_default_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_462: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_46, [1, 512, 768]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_463: "f32[512, 768]" = torch.ops.aten.reshape.default(view_462, [512, 768]);  view_462 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_463, permute_360);  permute_360 = None
    permute_361: "f32[768, 512]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[768]" = torch.ops.aten.reshape.default(sum_125, [768]);  sum_125 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_465: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_304, view_465);  mul_304 = view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_41, [0, 2, 1, 3]);  permute_default_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_466: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_364, [1, 512, 768]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_467: "f32[512, 768]" = torch.ops.aten.reshape.default(view_466, [512, 768]);  view_466 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_467, permute_365);  permute_365 = None
    permute_366: "f32[768, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_366, view_110);  permute_366 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.reshape.default(sum_126, [768]);  sum_126 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_469: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_148: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_147, view_469);  add_147 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_470: "f32[512, 768]" = torch.ops.aten.reshape.default(view_461, [512, 768]);  view_461 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_470, permute_369);  permute_369 = None
    permute_370: "f32[768, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[768]" = torch.ops.aten.reshape.default(sum_127, [768]);  sum_127 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_472: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_86, [1, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_148, view_472);  add_148 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_313: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_84);  primals_84 = None
    mul_314: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, 768)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True)
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_313, mul_37);  mul_313 = None
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True);  mul_315 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_37, sum_129);  sum_129 = None
    sub_95: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_314, sum_128);  mul_314 = sum_128 = None
    sub_96: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_316);  sub_95 = mul_316 = None
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_48, sub_96);  div_48 = sub_96 = None
    mul_318: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_37);  mul_37 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 1]);  mul_318 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_317, mul_319);  mul_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_320, [512, 768]);  mul_320 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_473, permute_373);  permute_373 = None
    permute_374: "f32[768, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[768]" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_475: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_322: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_42, 0.5);  add_42 = None
    mul_323: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_324: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_323, -0.5);  mul_323 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_324);  mul_324 = None
    mul_325: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_326: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_325);  view_107 = mul_325 = None
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_322, mul_326);  mul_322 = mul_326 = None
    mul_327: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_475, add_151);  view_475 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_327, [512, 3072]);  mul_327 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_476, permute_377);  permute_377 = None
    permute_378: "f32[3072, 512]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_133: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[3072]" = torch.ops.aten.reshape.default(sum_133, [3072]);  sum_133 = None
    permute_380: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_317, view_478);  mul_317 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_78);  primals_78 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, 768)
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_32);  mul_329 = None
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_32, sum_135);  sum_135 = None
    sub_98: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_330, sum_134);  mul_330 = sum_134 = None
    sub_99: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_332);  sub_98 = mul_332 = None
    mul_333: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_99);  div_49 = sub_99 = None
    mul_334: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_32);  mul_32 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_335: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_336: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_335);  mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_479: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_336, [512, 768]);  mul_336 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_479, permute_381);  permute_381 = None
    permute_382: "f32[768, 512]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_481: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_482: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_481, [1, 512, 12, 64]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
    
    # No stacktrace found for following nodes
    view_default_90: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_385, [12, 512, 64]);  permute_385 = None
    bmm_default_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_43, view_default_90);  permute_default_43 = None
    view_default_91: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_44, [1, 12, 512, 64]);  bmm_default_44 = None
    bmm_default_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_90, permute_default_44);  view_default_90 = permute_default_44 = None
    view_default_92: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_45, [1, 12, 512, 512]);  bmm_default_45 = None
    mul_tensor_29: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_92, mul_tensor_28);  view_default_92 = mul_tensor_28 = None
    mul_tensor_30: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_29, alias_default_15);  mul_tensor_29 = None
    sum_dim_int_list_15: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [-1], True)
    mul_tensor_31: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_15, sum_dim_int_list_15);  alias_default_15 = sum_dim_int_list_15 = None
    sub_tensor_15: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
    view_default_93: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_15, [12, 512, 512]);  sub_tensor_15 = None
    bmm_default_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_45, view_default_93);  permute_default_45 = None
    view_default_94: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_46, [1, 12, 64, 512]);  bmm_default_46 = None
    mul_scalar_30: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_94, 0.3535533905932738);  view_default_94 = None
    permute_default_47: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_30, [0, 1, 3, 2]);  mul_scalar_30 = None
    bmm_default_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_93, permute_default_46);  view_default_93 = permute_default_46 = None
    view_default_95: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_47, [1, 12, 512, 64]);  bmm_default_47 = None
    mul_scalar_31: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_95, 0.3535533905932738);  view_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_31, [0, 2, 1, 3]);  mul_scalar_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_489: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_50, [1, 512, 768]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_91, [0, 2, 1, 3]);  view_default_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_490: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_51, [1, 512, 768]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_491: "f32[512, 768]" = torch.ops.aten.reshape.default(view_490, [512, 768]);  view_490 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_491, permute_393);  permute_393 = None
    permute_394: "f32[768, 512]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[768]" = torch.ops.aten.reshape.default(sum_140, [768]);  sum_140 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_493: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_94, [1, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_333, view_493);  mul_333 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_47, [0, 2, 1, 3]);  permute_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_494: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_397, [1, 512, 768]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_495: "f32[512, 768]" = torch.ops.aten.reshape.default(view_494, [512, 768]);  view_494 = None
    mm_96: "f32[512, 768]" = torch.ops.aten.mm.default(view_495, permute_398);  permute_398 = None
    permute_399: "f32[768, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_88);  permute_399 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[768]" = torch.ops.aten.reshape.default(sum_141, [768]);  sum_141 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_497: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_96, [1, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_154: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_153, view_497);  add_153 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_498: "f32[512, 768]" = torch.ops.aten.reshape.default(view_489, [512, 768]);  view_489 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_498, permute_402);  permute_402 = None
    permute_403: "f32[768, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[768]" = torch.ops.aten.reshape.default(sum_142, [768]);  sum_142 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_500: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_154, view_500);  add_154 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_342: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_68);  primals_68 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_30);  mul_342 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_144);  sum_144 = None
    sub_102: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_143);  mul_343 = sum_143 = None
    sub_103: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_345);  sub_102 = mul_345 = None
    mul_346: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_103);  div_51 = sub_103 = None
    mul_347: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_30);  mul_30 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_348: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_346, mul_348);  mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_349, [512, 768]);  mul_349 = None
    mm_100: "f32[512, 3072]" = torch.ops.aten.mm.default(view_501, permute_406);  permute_406 = None
    permute_407: "f32[768, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
    permute_409: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_503: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_100, [1, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_351: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_34, 0.5);  add_34 = None
    mul_352: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_353: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_352, -0.5);  mul_352 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_353);  mul_353 = None
    mul_354: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_355: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_354);  view_85 = mul_354 = None
    add_157: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_351, mul_355);  mul_351 = mul_355 = None
    mul_356: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_503, add_157);  view_503 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_356, [512, 3072]);  mul_356 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_504, permute_410);  permute_410 = None
    permute_411: "f32[3072, 512]" = torch.ops.aten.permute.default(view_504, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
    view_505: "f32[3072]" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
    permute_413: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_506: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_102, [1, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_158: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_346, view_506);  mul_346 = view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_358: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_62);  primals_62 = None
    mul_359: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_358, 768)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
    mul_360: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_358, mul_25);  mul_358 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
    mul_361: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_150);  sum_150 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_359, sum_149);  mul_359 = sum_149 = None
    sub_106: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_361);  sub_105 = mul_361 = None
    mul_362: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
    mul_363: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_25);  mul_25 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_364: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_365: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_362, mul_364);  mul_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_507: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_365, [512, 768]);  mul_365 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_507, permute_414);  permute_414 = None
    permute_415: "f32[768, 512]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_509: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_104, [1, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_510: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_509, [1, 512, 12, 64]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
    
    # No stacktrace found for following nodes
    view_default_102: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_418, [12, 512, 64]);  permute_418 = None
    bmm_default_50: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_49, view_default_102);  permute_default_49 = None
    view_default_103: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_50, [1, 12, 512, 64]);  bmm_default_50 = None
    bmm_default_51: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_102, permute_default_50);  view_default_102 = permute_default_50 = None
    view_default_104: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_51, [1, 12, 512, 512]);  bmm_default_51 = None
    mul_tensor_33: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_104, mul_tensor_32);  view_default_104 = mul_tensor_32 = None
    mul_tensor_34: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_33, alias_default_17);  mul_tensor_33 = None
    sum_dim_int_list_17: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [-1], True)
    mul_tensor_35: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_17, sum_dim_int_list_17);  alias_default_17 = sum_dim_int_list_17 = None
    sub_tensor_17: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
    view_default_105: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_17, [12, 512, 512]);  sub_tensor_17 = None
    bmm_default_52: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_51, view_default_105);  permute_default_51 = None
    view_default_106: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_52, [1, 12, 64, 512]);  bmm_default_52 = None
    mul_scalar_34: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_106, 0.3535533905932738);  view_default_106 = None
    permute_default_53: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_34, [0, 1, 3, 2]);  mul_scalar_34 = None
    bmm_default_53: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_105, permute_default_52);  view_default_105 = permute_default_52 = None
    view_default_107: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_53, [1, 12, 512, 64]);  bmm_default_53 = None
    mul_scalar_35: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_107, 0.3535533905932738);  view_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_35, [0, 2, 1, 3]);  mul_scalar_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_517: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_55, [1, 512, 768]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_103, [0, 2, 1, 3]);  view_default_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_518: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_56, [1, 512, 768]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_519: "f32[512, 768]" = torch.ops.aten.reshape.default(view_518, [512, 768]);  view_518 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_519, permute_426);  permute_426 = None
    permute_427: "f32[768, 512]" = torch.ops.aten.permute.default(view_519, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
    view_520: "f32[768]" = torch.ops.aten.reshape.default(sum_155, [768]);  sum_155 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_521: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_362, view_521);  mul_362 = view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_53, [0, 2, 1, 3]);  permute_default_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_522: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_430, [1, 512, 768]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_523: "f32[512, 768]" = torch.ops.aten.reshape.default(view_522, [512, 768]);  view_522 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_523, permute_431);  permute_431 = None
    permute_432: "f32[768, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_66);  permute_432 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[768]" = torch.ops.aten.reshape.default(sum_156, [768]);  sum_156 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_525: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_160: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_525);  add_159 = view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_526: "f32[512, 768]" = torch.ops.aten.reshape.default(view_517, [512, 768]);  view_517 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_526, permute_435);  permute_435 = None
    permute_436: "f32[768, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[768]" = torch.ops.aten.reshape.default(sum_157, [768]);  sum_157 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_110, [1, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_161: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_160, view_528);  add_160 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_371: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_52);  primals_52 = None
    mul_372: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, 768)
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True)
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_371, mul_23);  mul_371 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [2], True);  mul_373 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_23, sum_159);  sum_159 = None
    sub_109: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_372, sum_158);  mul_372 = sum_158 = None
    sub_110: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_374);  sub_109 = mul_374 = None
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
    mul_376: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_23);  mul_23 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1]);  mul_376 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_375, mul_377);  mul_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_378, [512, 768]);  mul_378 = None
    mm_112: "f32[512, 3072]" = torch.ops.aten.mm.default(view_529, permute_439);  permute_439 = None
    permute_440: "f32[768, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[768]" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
    permute_442: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_531: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_112, [1, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_380: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_381, -0.5);  mul_381 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_382);  mul_382 = None
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_383);  view_63 = mul_383 = None
    add_163: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_380, mul_384);  mul_380 = mul_384 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_531, add_163);  view_531 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_532: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_385, [512, 3072]);  mul_385 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_532, permute_443);  permute_443 = None
    permute_444: "f32[3072, 512]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_163: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[3072]" = torch.ops.aten.reshape.default(sum_163, [3072]);  sum_163 = None
    permute_446: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_534: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_114, [1, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_164: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_375, view_534);  mul_375 = view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_46);  primals_46 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, 768)
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True)
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_387, mul_18);  mul_387 = None
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    mul_390: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_18, sum_165);  sum_165 = None
    sub_112: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_388, sum_164);  mul_388 = sum_164 = None
    sub_113: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_390);  sub_112 = mul_390 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_113);  div_55 = sub_113 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_18);  mul_18 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1]);  mul_392 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_394: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_393);  mul_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_535: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_394, [512, 768]);  mul_394 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_535, permute_447);  permute_447 = None
    permute_448: "f32[768, 512]" = torch.ops.aten.permute.default(view_535, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
    view_536: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_537: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_116, [1, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_538: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_537, [1, 512, 12, 64]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
    
    # No stacktrace found for following nodes
    view_default_114: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_451, [12, 512, 64]);  permute_451 = None
    bmm_default_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_55, view_default_114);  permute_default_55 = None
    view_default_115: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_56, [1, 12, 512, 64]);  bmm_default_56 = None
    bmm_default_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_114, permute_default_56);  view_default_114 = permute_default_56 = None
    view_default_116: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_57, [1, 12, 512, 512]);  bmm_default_57 = None
    mul_tensor_37: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_116, mul_tensor_36);  view_default_116 = mul_tensor_36 = None
    mul_tensor_38: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, alias_default_19);  mul_tensor_37 = None
    sum_dim_int_list_19: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [-1], True)
    mul_tensor_39: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_19, sum_dim_int_list_19);  alias_default_19 = sum_dim_int_list_19 = None
    sub_tensor_19: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
    view_default_117: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_19, [12, 512, 512]);  sub_tensor_19 = None
    bmm_default_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_57, view_default_117);  permute_default_57 = None
    view_default_118: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_58, [1, 12, 64, 512]);  bmm_default_58 = None
    mul_scalar_38: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_118, 0.3535533905932738);  view_default_118 = None
    permute_default_59: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_38, [0, 1, 3, 2]);  mul_scalar_38 = None
    bmm_default_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_117, permute_default_58);  view_default_117 = permute_default_58 = None
    view_default_119: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_59, [1, 12, 512, 64]);  bmm_default_59 = None
    mul_scalar_39: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_119, 0.3535533905932738);  view_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_39, [0, 2, 1, 3]);  mul_scalar_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_545: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_60, [1, 512, 768]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_115, [0, 2, 1, 3]);  view_default_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_546: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_61, [1, 512, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_547: "f32[512, 768]" = torch.ops.aten.reshape.default(view_546, [512, 768]);  view_546 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_547, permute_459);  permute_459 = None
    permute_460: "f32[768, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_549: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_118, [1, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_391, view_549);  mul_391 = view_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_59, [0, 2, 1, 3]);  permute_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_550: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_463, [1, 512, 768]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_551: "f32[512, 768]" = torch.ops.aten.reshape.default(view_550, [512, 768]);  view_550 = None
    mm_120: "f32[512, 768]" = torch.ops.aten.mm.default(view_551, permute_464);  permute_464 = None
    permute_465: "f32[768, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[768]" = torch.ops.aten.reshape.default(sum_171, [768]);  sum_171 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_553: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_120, [1, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_553);  add_165 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_554: "f32[512, 768]" = torch.ops.aten.reshape.default(view_545, [512, 768]);  view_545 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_554, permute_468);  permute_468 = None
    permute_469: "f32[768, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[768]" = torch.ops.aten.reshape.default(sum_172, [768]);  sum_172 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_556: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_122, [1, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_166, view_556);  add_166 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_400: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_36);  primals_36 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_400, 768)
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True)
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_400, mul_16);  mul_400 = None
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True);  mul_402 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_16, sum_174);  sum_174 = None
    sub_116: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_401, sum_173);  mul_401 = sum_173 = None
    sub_117: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_403);  sub_116 = mul_403 = None
    mul_404: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_117);  div_57 = sub_117 = None
    mul_405: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_16);  mul_16 = None
    sum_175: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 1]);  mul_405 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_406: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_407: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_404, mul_406);  mul_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_407, [512, 768]);  mul_407 = None
    mm_124: "f32[512, 3072]" = torch.ops.aten.mm.default(view_557, permute_472);  permute_472 = None
    permute_473: "f32[768, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[768]" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
    permute_475: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_559: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [1, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_411: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_410, -0.5);  mul_410 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_411);  mul_411 = None
    mul_412: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_413: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_412);  view_41 = mul_412 = None
    add_169: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_409, mul_413);  mul_409 = mul_413 = None
    mul_414: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_559, add_169);  view_559 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_560: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_414, [512, 3072]);  mul_414 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_476);  permute_476 = None
    permute_477: "f32[3072, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_178: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[3072]" = torch.ops.aten.reshape.default(sum_178, [3072]);  sum_178 = None
    permute_479: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_562: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_126, [1, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_170: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_404, view_562);  mul_404 = view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_30);  primals_30 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_11);  mul_416 = None
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_11, sum_180);  sum_180 = None
    sub_119: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_179);  mul_417 = sum_179 = None
    sub_120: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_119, mul_419);  sub_119 = mul_419 = None
    mul_420: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_120);  div_58 = sub_120 = None
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_11);  mul_11 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_182: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_422: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_563: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_423, [512, 768]);  mul_423 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_563, permute_480);  permute_480 = None
    permute_481: "f32[768, 512]" = torch.ops.aten.permute.default(view_563, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
    view_564: "f32[768]" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_565: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_128, [1, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_566: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_565, [1, 512, 12, 64]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
    
    # No stacktrace found for following nodes
    view_default_126: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_484, [12, 512, 64]);  permute_484 = None
    bmm_default_62: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_61, view_default_126);  permute_default_61 = None
    view_default_127: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_62, [1, 12, 512, 64]);  bmm_default_62 = None
    bmm_default_63: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_126, permute_default_62);  view_default_126 = permute_default_62 = None
    view_default_128: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_63, [1, 12, 512, 512]);  bmm_default_63 = None
    mul_tensor_41: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_128, mul_tensor_40);  view_default_128 = mul_tensor_40 = None
    mul_tensor_42: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_41, alias_default_21);  mul_tensor_41 = None
    sum_dim_int_list_21: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [-1], True)
    mul_tensor_43: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_21, sum_dim_int_list_21);  alias_default_21 = sum_dim_int_list_21 = None
    sub_tensor_21: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
    view_default_129: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_21, [12, 512, 512]);  sub_tensor_21 = None
    bmm_default_64: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_63, view_default_129);  permute_default_63 = None
    view_default_130: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_64, [1, 12, 64, 512]);  bmm_default_64 = None
    mul_scalar_42: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_130, 0.3535533905932738);  view_default_130 = None
    permute_default_65: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_42, [0, 1, 3, 2]);  mul_scalar_42 = None
    bmm_default_65: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_129, permute_default_64);  view_default_129 = permute_default_64 = None
    view_default_131: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_65, [1, 12, 512, 64]);  bmm_default_65 = None
    mul_scalar_43: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_131, 0.3535533905932738);  view_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_43, [0, 2, 1, 3]);  mul_scalar_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_573: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_65, [1, 512, 768]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_127, [0, 2, 1, 3]);  view_default_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_574: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_66, [1, 512, 768]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_575: "f32[512, 768]" = torch.ops.aten.reshape.default(view_574, [512, 768]);  view_574 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_575, permute_492);  permute_492 = None
    permute_493: "f32[768, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.reshape.default(sum_185, [768]);  sum_185 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_577: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_171: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_420, view_577);  mul_420 = view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_65, [0, 2, 1, 3]);  permute_default_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_578: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_496, [1, 512, 768]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_579: "f32[512, 768]" = torch.ops.aten.reshape.default(view_578, [512, 768]);  view_578 = None
    mm_132: "f32[512, 768]" = torch.ops.aten.mm.default(view_579, permute_497);  permute_497 = None
    permute_498: "f32[768, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_498, view_22);  permute_498 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[768]" = torch.ops.aten.reshape.default(sum_186, [768]);  sum_186 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_581: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_171, view_581);  add_171 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_582: "f32[512, 768]" = torch.ops.aten.reshape.default(view_573, [512, 768]);  view_573 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_582, permute_501);  permute_501 = None
    permute_502: "f32[768, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_584: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_172, view_584);  add_172 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:381, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_429: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_20);  primals_20 = None
    mul_430: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, 768)
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True)
    mul_431: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_429, mul_9);  mul_429 = None
    sum_189: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
    mul_432: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_189);  sum_189 = None
    sub_123: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_430, sum_188);  mul_430 = sum_188 = None
    sub_124: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_432);  sub_123 = mul_432 = None
    mul_433: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_60, sub_124);  div_60 = sub_124 = None
    mul_434: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_9);  mul_9 = None
    sum_190: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 1]);  mul_434 = None
    sum_191: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:380, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_37: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_435: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_436: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_433, mul_435);  mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:379, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_436, [512, 768]);  mul_436 = None
    mm_136: "f32[512, 3072]" = torch.ops.aten.mm.default(view_585, permute_505);  permute_505 = None
    permute_506: "f32[768, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[768]" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
    permute_508: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_587: "f32[1, 512, 3072]" = torch.ops.aten.reshape.default(mm_136, [1, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_438: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
    mul_439: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_440: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_439, -0.5);  mul_439 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_440);  mul_440 = None
    mul_441: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_442: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_441);  view_19 = mul_441 = None
    add_175: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_438, mul_442);  mul_438 = mul_442 = None
    mul_443: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_587, add_175);  view_587 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[512, 3072]" = torch.ops.aten.reshape.default(mul_443, [512, 3072]);  mul_443 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_588, permute_509);  permute_509 = None
    permute_510: "f32[3072, 512]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_193: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[3072]" = torch.ops.aten.reshape.default(sum_193, [3072]);  sum_193 = None
    permute_512: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_590: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_138, [1, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:365, code: hidden_states = self.dense(hidden_states)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_433, view_590);  mul_433 = view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:300, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_445: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_14);  primals_14 = None
    mul_446: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_445, 768)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
    mul_447: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_445, mul_4);  mul_445 = None
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
    mul_448: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_4, sum_195);  sum_195 = None
    sub_126: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_446, sum_194);  mul_446 = sum_194 = None
    sub_127: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_126, mul_448);  sub_126 = mul_448 = None
    mul_449: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_127);  div_61 = sub_127 = None
    mul_450: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_4);  mul_4 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:299, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_38: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_451: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_452: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_449, mul_451);  mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:298, code: hidden_states = self.dense(hidden_states)
    view_591: "f32[512, 768]" = torch.ops.aten.reshape.default(mul_452, [512, 768]);  mul_452 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_591, permute_513);  permute_513 = None
    permute_514: "f32[768, 512]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
    permute_515: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_593: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_140, [1, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:280, code: context_layer = context_layer.view(new_context_layer_shape)
    view_594: "f32[1, 512, 12, 64]" = torch.ops.aten.reshape.default(view_593, [1, 512, 12, 64]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:278, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_594, [0, 2, 1, 3]);  view_594 = None
    
    # No stacktrace found for following nodes
    view_default_138: "f32[12, 512, 64]" = torch.ops.aten.reshape.default(permute_517, [12, 512, 64]);  permute_517 = None
    bmm_default_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_default_67, view_default_138);  permute_default_67 = None
    view_default_139: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_68, [1, 12, 512, 64]);  bmm_default_68 = None
    bmm_default_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_default_138, permute_default_68);  view_default_138 = permute_default_68 = None
    view_default_140: "f32[1, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_69, [1, 12, 512, 512]);  bmm_default_69 = None
    mul_tensor_45: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_140, mul_tensor_44);  view_default_140 = mul_tensor_44 = None
    mul_tensor_46: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_45, alias_default_23);  mul_tensor_45 = None
    sum_dim_int_list_23: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [-1], True)
    mul_tensor_47: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_23, sum_dim_int_list_23);  alias_default_23 = sum_dim_int_list_23 = None
    sub_tensor_23: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_46, mul_tensor_47);  mul_tensor_46 = mul_tensor_47 = None
    view_default_141: "f32[12, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_23, [12, 512, 512]);  sub_tensor_23 = None
    bmm_default_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_default_69, view_default_141);  permute_default_69 = None
    view_default_142: "f32[1, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_70, [1, 12, 64, 512]);  bmm_default_70 = None
    mul_scalar_46: "f32[1, 12, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_142, 0.3535533905932738);  view_default_142 = None
    permute_default_71: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_46, [0, 1, 3, 2]);  mul_scalar_46 = None
    bmm_default_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_default_141, permute_default_70);  view_default_141 = permute_default_70 = None
    view_default_143: "f32[1, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_71, [1, 12, 512, 64]);  bmm_default_71 = None
    mul_scalar_47: "f32[1, 12, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_143, 0.3535533905932738);  view_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(mul_scalar_47, [0, 2, 1, 3]);  mul_scalar_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_601: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_70, [1, 512, 768]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(view_default_139, [0, 2, 1, 3]);  view_default_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_602: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(clone_71, [1, 512, 768]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_603: "f32[512, 768]" = torch.ops.aten.reshape.default(view_602, [512, 768]);  view_602 = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_603, permute_525);  permute_525 = None
    permute_526: "f32[768, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_605: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:220, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_449, view_605);  mul_449 = view_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:185, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[1, 512, 12, 64]" = torch.ops.aten.permute.default(permute_default_71, [0, 2, 1, 3]);  permute_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:184, code: x = x.view(new_x_shape)
    view_606: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(permute_529, [1, 512, 768]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_607: "f32[512, 768]" = torch.ops.aten.reshape.default(view_606, [512, 768]);  view_606 = None
    mm_144: "f32[512, 768]" = torch.ops.aten.mm.default(view_607, permute_530);  permute_530 = None
    permute_531: "f32[768, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[768]" = torch.ops.aten.reshape.default(sum_201, [768]);  sum_201 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_609: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:219, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_178: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_177, view_609);  add_177 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    view_610: "f32[512, 768]" = torch.ops.aten.reshape.default(view_601, [512, 768]);  view_601 = None
    mm_146: "f32[512, 768]" = torch.ops.aten.mm.default(view_610, permute_534);  permute_534 = None
    permute_535: "f32[768, 512]" = torch.ops.aten.permute.default(view_610, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_610, [0], True);  view_610 = None
    view_611: "f32[768]" = torch.ops.aten.reshape.default(sum_202, [768]);  sum_202 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_612: "f32[1, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [1, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:197, code: mixed_query_layer = self.query(hidden_states)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_178, view_612);  add_178 = view_612 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:133, code: embeddings = self.dropout(embeddings)
    convert_element_type_40: "f32[1, 512, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_457: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
    mul_458: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_457);  add_179 = mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:132, code: embeddings = self.LayerNorm(embeddings)
    mul_460: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_458, primals_4);  primals_4 = None
    mul_461: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_460, 768)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [2], True)
    mul_462: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_460, mul_2);  mul_460 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
    mul_463: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_2, sum_204);  sum_204 = None
    sub_130: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_461, sum_203);  mul_461 = sum_203 = None
    sub_131: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_130, mul_463);  sub_130 = mul_463 = None
    mul_464: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_63, sub_131);  div_63 = sub_131 = None
    mul_465: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_458, mul_2);  mul_2 = None
    sum_205: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 1]);  mul_465 = None
    sum_206: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 1]);  mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:130, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(add_1, 0)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_2, mul_464);  unsqueeze_4 = None
    full_default_10: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_10, [add_1], where_4, True);  full_default_10 = add_1 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:126, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_5: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_5, full_default_2, mul_464);  unsqueeze_5 = None
    full_default_12: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_12, [expand], where_5, True);  full_default_12 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/roberta/modeling_roberta.py:125, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_206, 0)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_6: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_6, full_default_2, mul_464);  unsqueeze_6 = full_default_2 = mul_464 = None
    full_default_14: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[50265, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_14, [primals_206], where_6, True);  full_default_14 = primals_206 = where_6 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_537, view_611, permute_533, view_608, permute_528, view_604, permute_516, view_592, sum_196, sum_197, permute_512, view_589, permute_508, view_586, sum_190, sum_191, permute_504, view_583, permute_500, view_580, permute_495, view_576, permute_483, view_564, sum_181, sum_182, permute_479, view_561, permute_475, view_558, sum_175, sum_176, permute_471, view_555, permute_467, view_552, permute_462, view_548, permute_450, view_536, sum_166, sum_167, permute_446, view_533, permute_442, view_530, sum_160, sum_161, permute_438, view_527, permute_434, view_524, permute_429, view_520, permute_417, view_508, sum_151, sum_152, permute_413, view_505, permute_409, view_502, sum_145, sum_146, permute_405, view_499, permute_401, view_496, permute_396, view_492, permute_384, view_480, sum_136, sum_137, permute_380, view_477, permute_376, view_474, sum_130, sum_131, permute_372, view_471, permute_368, view_468, permute_363, view_464, permute_351, view_452, sum_121, sum_122, permute_347, view_449, permute_343, view_446, sum_115, sum_116, permute_339, view_443, permute_335, view_440, permute_330, view_436, permute_318, view_424, sum_106, sum_107, permute_314, view_421, permute_310, view_418, sum_100, sum_101, permute_306, view_415, permute_302, view_412, permute_297, view_408, permute_285, view_396, sum_91, sum_92, permute_281, view_393, permute_277, view_390, sum_85, sum_86, permute_273, view_387, permute_269, view_384, permute_264, view_380, permute_252, view_368, sum_76, sum_77, permute_248, view_365, permute_244, view_362, sum_70, sum_71, permute_240, view_359, permute_236, view_356, permute_231, view_352, permute_219, view_340, sum_61, sum_62, permute_215, view_337, permute_211, view_334, sum_55, sum_56, permute_207, view_331, permute_203, view_328, permute_198, view_324, permute_186, view_312, sum_46, sum_47, permute_182, view_309, permute_178, view_306, sum_40, sum_41, permute_174, view_303, permute_170, view_300, permute_165, view_296, permute_153, view_284, sum_31, sum_32, permute_149, view_281, permute_145, view_278, sum_25, sum_26, permute_141, view_275, sum_20, sum_21, permute_137, view_272, None, None, None]
    