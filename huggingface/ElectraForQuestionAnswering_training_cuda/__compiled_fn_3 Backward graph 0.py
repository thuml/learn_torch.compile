from __future__ import annotations



def forward(self, primals_4: "f32[128]", primals_16: "f32[256]", primals_22: "f32[256]", primals_32: "f32[256]", primals_38: "f32[256]", primals_48: "f32[256]", primals_54: "f32[256]", primals_64: "f32[256]", primals_70: "f32[256]", primals_80: "f32[256]", primals_86: "f32[256]", primals_96: "f32[256]", primals_102: "f32[256]", primals_112: "f32[256]", primals_118: "f32[256]", primals_128: "f32[256]", primals_134: "f32[256]", primals_144: "f32[256]", primals_150: "f32[256]", primals_160: "f32[256]", primals_166: "f32[256]", primals_176: "f32[256]", primals_182: "f32[256]", primals_192: "f32[256]", primals_198: "f32[256]", primals_204: "i64[1, 512]", expand: "i64[1, 512]", slice_4: "i64[1, 512]", mul_1: "f32[1, 512, 128]", getitem_3: "b8[1, 512, 128]", view: "f32[512, 128]", view_2: "f32[512, 256]", clone_default_33: "f32[1, 4, 512, 64]", clone_default_34: "f32[1, 4, 512, 64]", clone_default_35: "f32[1, 4, 512, 64]", getitem_204: "f32[1, 4, 512]", getitem_205: "i64[]", getitem_206: "i64[]", alias_default_23: "f32[1, 4, 512, 64]", view_18: "f32[512, 256]", getitem_7: "b8[1, 512, 256]", mul_3: "f32[1, 512, 256]", view_20: "f32[512, 256]", addmm_5: "f32[512, 1024]", view_22: "f32[512, 1024]", getitem_11: "b8[1, 512, 256]", mul_8: "f32[1, 512, 256]", view_24: "f32[512, 256]", clone_default_30: "f32[1, 4, 512, 64]", clone_default_31: "f32[1, 4, 512, 64]", clone_default_32: "f32[1, 4, 512, 64]", getitem_197: "f32[1, 4, 512]", getitem_198: "i64[]", getitem_199: "i64[]", alias_default_21: "f32[1, 4, 512, 64]", view_40: "f32[512, 256]", getitem_17: "b8[1, 512, 256]", mul_10: "f32[1, 512, 256]", view_42: "f32[512, 256]", addmm_11: "f32[512, 1024]", view_44: "f32[512, 1024]", getitem_21: "b8[1, 512, 256]", mul_15: "f32[1, 512, 256]", view_46: "f32[512, 256]", clone_default_27: "f32[1, 4, 512, 64]", clone_default_28: "f32[1, 4, 512, 64]", clone_default_29: "f32[1, 4, 512, 64]", getitem_190: "f32[1, 4, 512]", getitem_191: "i64[]", getitem_192: "i64[]", alias_default_19: "f32[1, 4, 512, 64]", view_62: "f32[512, 256]", getitem_27: "b8[1, 512, 256]", mul_17: "f32[1, 512, 256]", view_64: "f32[512, 256]", addmm_17: "f32[512, 1024]", view_66: "f32[512, 1024]", getitem_31: "b8[1, 512, 256]", mul_22: "f32[1, 512, 256]", view_68: "f32[512, 256]", clone_default_24: "f32[1, 4, 512, 64]", clone_default_25: "f32[1, 4, 512, 64]", clone_default_26: "f32[1, 4, 512, 64]", getitem_183: "f32[1, 4, 512]", getitem_184: "i64[]", getitem_185: "i64[]", alias_default_17: "f32[1, 4, 512, 64]", view_84: "f32[512, 256]", getitem_37: "b8[1, 512, 256]", mul_24: "f32[1, 512, 256]", view_86: "f32[512, 256]", addmm_23: "f32[512, 1024]", view_88: "f32[512, 1024]", getitem_41: "b8[1, 512, 256]", mul_29: "f32[1, 512, 256]", view_90: "f32[512, 256]", clone_default_21: "f32[1, 4, 512, 64]", clone_default_22: "f32[1, 4, 512, 64]", clone_default_23: "f32[1, 4, 512, 64]", getitem_176: "f32[1, 4, 512]", getitem_177: "i64[]", getitem_178: "i64[]", alias_default_15: "f32[1, 4, 512, 64]", view_106: "f32[512, 256]", getitem_47: "b8[1, 512, 256]", mul_31: "f32[1, 512, 256]", view_108: "f32[512, 256]", addmm_29: "f32[512, 1024]", view_110: "f32[512, 1024]", getitem_51: "b8[1, 512, 256]", mul_36: "f32[1, 512, 256]", view_112: "f32[512, 256]", clone_default_18: "f32[1, 4, 512, 64]", clone_default_19: "f32[1, 4, 512, 64]", clone_default_20: "f32[1, 4, 512, 64]", getitem_169: "f32[1, 4, 512]", getitem_170: "i64[]", getitem_171: "i64[]", alias_default_13: "f32[1, 4, 512, 64]", view_128: "f32[512, 256]", getitem_57: "b8[1, 512, 256]", mul_38: "f32[1, 512, 256]", view_130: "f32[512, 256]", addmm_35: "f32[512, 1024]", view_132: "f32[512, 1024]", getitem_61: "b8[1, 512, 256]", mul_43: "f32[1, 512, 256]", view_134: "f32[512, 256]", clone_default_15: "f32[1, 4, 512, 64]", clone_default_16: "f32[1, 4, 512, 64]", clone_default_17: "f32[1, 4, 512, 64]", getitem_162: "f32[1, 4, 512]", getitem_163: "i64[]", getitem_164: "i64[]", alias_default_11: "f32[1, 4, 512, 64]", view_150: "f32[512, 256]", getitem_67: "b8[1, 512, 256]", mul_45: "f32[1, 512, 256]", view_152: "f32[512, 256]", addmm_41: "f32[512, 1024]", view_154: "f32[512, 1024]", getitem_71: "b8[1, 512, 256]", mul_50: "f32[1, 512, 256]", view_156: "f32[512, 256]", clone_default_12: "f32[1, 4, 512, 64]", clone_default_13: "f32[1, 4, 512, 64]", clone_default_14: "f32[1, 4, 512, 64]", getitem_155: "f32[1, 4, 512]", getitem_156: "i64[]", getitem_157: "i64[]", alias_default_9: "f32[1, 4, 512, 64]", view_172: "f32[512, 256]", getitem_77: "b8[1, 512, 256]", mul_52: "f32[1, 512, 256]", view_174: "f32[512, 256]", addmm_47: "f32[512, 1024]", view_176: "f32[512, 1024]", getitem_81: "b8[1, 512, 256]", mul_57: "f32[1, 512, 256]", view_178: "f32[512, 256]", clone_default_9: "f32[1, 4, 512, 64]", clone_default_10: "f32[1, 4, 512, 64]", clone_default_11: "f32[1, 4, 512, 64]", getitem_148: "f32[1, 4, 512]", getitem_149: "i64[]", getitem_150: "i64[]", alias_default_7: "f32[1, 4, 512, 64]", view_194: "f32[512, 256]", getitem_87: "b8[1, 512, 256]", mul_59: "f32[1, 512, 256]", view_196: "f32[512, 256]", addmm_53: "f32[512, 1024]", view_198: "f32[512, 1024]", getitem_91: "b8[1, 512, 256]", mul_64: "f32[1, 512, 256]", view_200: "f32[512, 256]", clone_default_6: "f32[1, 4, 512, 64]", clone_default_7: "f32[1, 4, 512, 64]", clone_default_8: "f32[1, 4, 512, 64]", getitem_141: "f32[1, 4, 512]", getitem_142: "i64[]", getitem_143: "i64[]", alias_default_5: "f32[1, 4, 512, 64]", view_216: "f32[512, 256]", getitem_97: "b8[1, 512, 256]", mul_66: "f32[1, 512, 256]", view_218: "f32[512, 256]", addmm_59: "f32[512, 1024]", view_220: "f32[512, 1024]", getitem_101: "b8[1, 512, 256]", mul_71: "f32[1, 512, 256]", view_222: "f32[512, 256]", clone_default_3: "f32[1, 4, 512, 64]", clone_default_4: "f32[1, 4, 512, 64]", clone_default_5: "f32[1, 4, 512, 64]", getitem_134: "f32[1, 4, 512]", getitem_135: "i64[]", getitem_136: "i64[]", alias_default_3: "f32[1, 4, 512, 64]", view_238: "f32[512, 256]", getitem_107: "b8[1, 512, 256]", mul_73: "f32[1, 512, 256]", view_240: "f32[512, 256]", addmm_65: "f32[512, 1024]", view_242: "f32[512, 1024]", getitem_111: "b8[1, 512, 256]", mul_78: "f32[1, 512, 256]", view_244: "f32[512, 256]", clone_default: "f32[1, 4, 512, 64]", clone_default_1: "f32[1, 4, 512, 64]", clone_default_2: "f32[1, 4, 512, 64]", getitem_127: "f32[1, 4, 512]", getitem_128: "i64[]", getitem_129: "i64[]", alias_default_1: "f32[1, 4, 512, 64]", view_260: "f32[512, 256]", getitem_117: "b8[1, 512, 256]", mul_80: "f32[1, 512, 256]", view_262: "f32[512, 256]", addmm_71: "f32[512, 1024]", view_264: "f32[512, 1024]", getitem_121: "b8[1, 512, 256]", mul_85: "f32[1, 512, 256]", view_266: "f32[512, 256]", sub_39: "f32[1, 512]", ne: "b8[1]", sub_41: "f32[1, 512]", ne_3: "b8[1]", ne_6: "b8[1, 1]", where_4: "i64[1, 1]", ne_8: "b8[1, 1]", where_6: "i64[1, 1]", permute_134: "f32[2, 256]", div_30: "f32[1, 512, 1]", permute_138: "f32[256, 1024]", permute_142: "f32[1024, 256]", div_31: "f32[1, 512, 1]", permute_146: "f32[256, 256]", permute_158: "f32[256, 256]", permute_163: "f32[256, 256]", permute_167: "f32[256, 256]", div_33: "f32[1, 512, 1]", permute_171: "f32[256, 1024]", permute_175: "f32[1024, 256]", div_34: "f32[1, 512, 1]", permute_179: "f32[256, 256]", permute_191: "f32[256, 256]", permute_196: "f32[256, 256]", permute_200: "f32[256, 256]", div_36: "f32[1, 512, 1]", permute_204: "f32[256, 1024]", permute_208: "f32[1024, 256]", div_37: "f32[1, 512, 1]", permute_212: "f32[256, 256]", permute_224: "f32[256, 256]", permute_229: "f32[256, 256]", permute_233: "f32[256, 256]", div_39: "f32[1, 512, 1]", permute_237: "f32[256, 1024]", permute_241: "f32[1024, 256]", div_40: "f32[1, 512, 1]", permute_245: "f32[256, 256]", permute_257: "f32[256, 256]", permute_262: "f32[256, 256]", permute_266: "f32[256, 256]", div_42: "f32[1, 512, 1]", permute_270: "f32[256, 1024]", permute_274: "f32[1024, 256]", div_43: "f32[1, 512, 1]", permute_278: "f32[256, 256]", permute_290: "f32[256, 256]", permute_295: "f32[256, 256]", permute_299: "f32[256, 256]", div_45: "f32[1, 512, 1]", permute_303: "f32[256, 1024]", permute_307: "f32[1024, 256]", div_46: "f32[1, 512, 1]", permute_311: "f32[256, 256]", permute_323: "f32[256, 256]", permute_328: "f32[256, 256]", permute_332: "f32[256, 256]", div_48: "f32[1, 512, 1]", permute_336: "f32[256, 1024]", permute_340: "f32[1024, 256]", div_49: "f32[1, 512, 1]", permute_344: "f32[256, 256]", permute_356: "f32[256, 256]", permute_361: "f32[256, 256]", permute_365: "f32[256, 256]", div_51: "f32[1, 512, 1]", permute_369: "f32[256, 1024]", permute_373: "f32[1024, 256]", div_52: "f32[1, 512, 1]", permute_377: "f32[256, 256]", permute_389: "f32[256, 256]", permute_394: "f32[256, 256]", permute_398: "f32[256, 256]", div_54: "f32[1, 512, 1]", permute_402: "f32[256, 1024]", permute_406: "f32[1024, 256]", div_55: "f32[1, 512, 1]", permute_410: "f32[256, 256]", permute_422: "f32[256, 256]", permute_427: "f32[256, 256]", permute_431: "f32[256, 256]", div_57: "f32[1, 512, 1]", permute_435: "f32[256, 1024]", permute_439: "f32[1024, 256]", div_58: "f32[1, 512, 1]", permute_443: "f32[256, 256]", permute_455: "f32[256, 256]", permute_460: "f32[256, 256]", permute_464: "f32[256, 256]", div_60: "f32[1, 512, 1]", permute_468: "f32[256, 1024]", permute_472: "f32[1024, 256]", div_61: "f32[1, 512, 1]", permute_476: "f32[256, 256]", permute_488: "f32[256, 256]", permute_493: "f32[256, 256]", permute_497: "f32[256, 256]", div_63: "f32[1, 512, 1]", permute_501: "f32[256, 1024]", permute_505: "f32[1024, 256]", div_64: "f32[1, 512, 1]", permute_509: "f32[256, 256]", permute_521: "f32[256, 256]", permute_526: "f32[256, 256]", permute_530: "f32[256, 256]", permute_534: "f32[256, 128]", div_66: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512]", tangents_3: "f32[1, 512]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_1: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
    erf_2: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_3: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_4: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_5: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
    erf_6: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
    erf_7: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_8: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
    erf_9: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
    erf_10: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_11: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    alias_12: "f32[1, 512]" = torch.ops.aten.alias.default(sub_39);  sub_39 = None
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    sum_14: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    alias_13: "f32[1, 512]" = torch.ops.aten.alias.default(sub_41);  sub_41 = None
    sum_17: "i64[]" = torch.ops.aten.sum.default(ne_3);  ne_3 = None
    convert_element_type_1: "f32[]" = torch.ops.prims.convert_element_type.default(sum_17, torch.float32);  sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1424, code: total_loss = (start_loss + end_loss) / 2
    div_27: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    div_28: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type_1);  convert_element_type_1 = None
    full_default_6: "f32[1, 512]" = torch.ops.aten.full.default([1, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    scatter: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_4, -1.0);  where_4 = None
    where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_6, div_28, full_default_2);  ne_6 = div_28 = None
    mul_87: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter, where_5);  scatter = where_5 = None
    alias_14: "f32[1, 512]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    exp_14: "f32[1, 512]" = torch.ops.aten.exp.default(alias_14);  alias_14 = None
    sum_19: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [1], True)
    mul_88: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_14, sum_19);  exp_14 = sum_19 = None
    sub_42: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1423, code: end_loss = loss_fct(end_logits, end_positions)
    add_101: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_3, sub_42);  tangents_3 = sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    div_29: "f32[]" = torch.ops.aten.div.Tensor(div_27, convert_element_type);  div_27 = convert_element_type = None
    scatter_1: "f32[1, 512]" = torch.ops.aten.scatter.value(full_default_6, 1, where_6, -1.0);  full_default_6 = where_6 = None
    where_7: "f32[1, 1]" = torch.ops.aten.where.self(ne_8, div_29, full_default_2);  ne_8 = div_29 = None
    mul_89: "f32[1, 512]" = torch.ops.aten.mul.Tensor(scatter_1, where_7);  scatter_1 = where_7 = None
    alias_15: "f32[1, 512]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    exp_15: "f32[1, 512]" = torch.ops.aten.exp.default(alias_15);  alias_15 = None
    sum_20: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [1], True)
    mul_90: "f32[1, 512]" = torch.ops.aten.mul.Tensor(exp_15, sum_20);  exp_15 = sum_20 = None
    sub_43: "f32[1, 512]" = torch.ops.aten.sub.Tensor(mul_89, mul_90);  mul_89 = mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1422, code: start_loss = loss_fct(start_logits, start_positions)
    add_102: "f32[1, 512]" = torch.ops.aten.add.Tensor(tangents_2, sub_43);  tangents_2 = sub_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_6: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_101, 2);  add_101 = None
    unsqueeze_7: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(add_102, 2);  add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1405, code: start_logits, end_logits = logits.split(1, dim=-1)
    cat: "f32[1, 512, 2]" = torch.ops.aten.cat.default([unsqueeze_7, unsqueeze_6], 2);  unsqueeze_7 = unsqueeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1404, code: logits = self.qa_outputs(sequence_output)
    view_268: "f32[512, 2]" = torch.ops.aten.view.default(cat, [512, 2]);  cat = None
    mm: "f32[512, 256]" = torch.ops.aten.mm.default(view_268, permute_134);  permute_134 = None
    permute_135: "f32[2, 512]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_1: "f32[2, 256]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[256, 2]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_21: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[2]" = torch.ops.aten.view.default(sum_21, [2]);  sum_21 = None
    permute_137: "f32[2, 256]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_270: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm, [1, 512, 256]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_92: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_270, primals_198);  primals_198 = None
    mul_93: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, 256)
    sum_22: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True)
    mul_94: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_92, mul_85);  mul_92 = None
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True);  mul_94 = None
    mul_95: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, sum_23);  sum_23 = None
    sub_45: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_93, sum_22);  mul_93 = sum_22 = None
    sub_46: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_45, mul_95);  sub_45 = mul_95 = None
    mul_96: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_46);  div_30 = sub_46 = None
    mul_97: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_270, mul_85);  mul_85 = None
    sum_24: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_97, [0, 1]);  mul_97 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_98: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_99: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_96, mul_98);  mul_98 = None
    clone_14: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_99, memory_format = torch.contiguous_format);  mul_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_271: "f32[512, 256]" = torch.ops.aten.view.default(clone_14, [512, 256]);  clone_14 = None
    mm_2: "f32[512, 1024]" = torch.ops.aten.mm.default(view_271, permute_138);  permute_138 = None
    permute_139: "f32[256, 512]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_3: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_26: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[256]" = torch.ops.aten.view.default(sum_26, [256]);  sum_26 = None
    permute_141: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_273: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_2, [1, 512, 1024]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_101: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_102: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_103: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_102, -0.5);  mul_102 = None
    exp_16: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_103);  mul_103 = None
    mul_104: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_105: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, mul_104);  view_263 = mul_104 = None
    add_104: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_101, mul_105);  mul_101 = mul_105 = None
    mul_106: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_273, add_104);  view_273 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_274: "f32[512, 1024]" = torch.ops.aten.view.default(mul_106, [512, 1024]);  mul_106 = None
    mm_4: "f32[512, 256]" = torch.ops.aten.mm.default(view_274, permute_142);  permute_142 = None
    permute_143: "f32[1024, 512]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_5: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[1024]" = torch.ops.aten.view.default(sum_27, [1024]);  sum_27 = None
    permute_145: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_276: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_4, [1, 512, 256]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_105: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_96, view_276);  mul_96 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_108: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_105, primals_192);  primals_192 = None
    mul_109: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, 256)
    sum_28: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_108, mul_80);  mul_108 = None
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, sum_29);  sum_29 = None
    sub_48: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_109, sum_28);  mul_109 = sum_28 = None
    sub_49: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_48, mul_111);  sub_48 = mul_111 = None
    mul_112: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_49);  div_31 = sub_49 = None
    mul_113: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_105, mul_80);  mul_80 = None
    sum_30: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_105, [0, 1]);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_3: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_114: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
    mul_115: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_112, mul_114);  mul_114 = None
    clone_15: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_115, memory_format = torch.contiguous_format);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[512, 256]" = torch.ops.aten.view.default(clone_15, [512, 256]);  clone_15 = None
    mm_6: "f32[512, 256]" = torch.ops.aten.mm.default(view_277, permute_146);  permute_146 = None
    permute_147: "f32[256, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_7: "f32[256, 256]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[256, 256]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_32: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[256]" = torch.ops.aten.view.default(sum_32, [256]);  sum_32 = None
    permute_149: "f32[256, 256]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_279: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_6, [1, 512, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_280: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_279, [1, 512, 4, 64]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_150, clone_default, clone_default_1, clone_default_2, None, alias_default_1, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_150 = clone_default = clone_default_1 = clone_default_2 = alias_default_1 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_130: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
    getitem_131: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
    getitem_132: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_156: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_130, [0, 2, 1, 3]);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_287: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_17, [1, 512, 256]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_157: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
    view_288: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_18, [1, 512, 256]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_289: "f32[512, 256]" = torch.ops.aten.view.default(view_288, [512, 256]);  view_288 = None
    mm_8: "f32[512, 256]" = torch.ops.aten.mm.default(view_289, permute_158);  permute_158 = None
    permute_159: "f32[256, 512]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_159, view_244);  permute_159 = None
    permute_160: "f32[256, 256]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_34: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[256]" = torch.ops.aten.view.default(sum_34, [256]);  sum_34 = None
    permute_161: "f32[256, 256]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_291: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_8, [1, 512, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_106: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_112, view_291);  mul_112 = view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_131, [0, 2, 1, 3]);  getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_292: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_162, [1, 512, 256]);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_293: "f32[512, 256]" = torch.ops.aten.view.default(view_292, [512, 256]);  view_292 = None
    mm_10: "f32[512, 256]" = torch.ops.aten.mm.default(view_293, permute_163);  permute_163 = None
    permute_164: "f32[256, 512]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_164, view_244);  permute_164 = None
    permute_165: "f32[256, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[256]" = torch.ops.aten.view.default(sum_35, [256]);  sum_35 = None
    permute_166: "f32[256, 256]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_295: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_10, [1, 512, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_107: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_106, view_295);  add_106 = view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_296: "f32[512, 256]" = torch.ops.aten.view.default(view_287, [512, 256]);  view_287 = None
    mm_12: "f32[512, 256]" = torch.ops.aten.mm.default(view_296, permute_167);  permute_167 = None
    permute_168: "f32[256, 512]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_13: "f32[256, 256]" = torch.ops.aten.mm.default(permute_168, view_244);  permute_168 = view_244 = None
    permute_169: "f32[256, 256]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[256]" = torch.ops.aten.view.default(sum_36, [256]);  sum_36 = None
    permute_170: "f32[256, 256]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_298: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_12, [1, 512, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_108: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_107, view_298);  add_107 = view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_121: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, primals_182);  primals_182 = None
    mul_122: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_121, 256)
    sum_37: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_121, mul_78);  mul_121 = None
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, sum_38);  sum_38 = None
    sub_52: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_122, sum_37);  mul_122 = sum_37 = None
    sub_53: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_52, mul_124);  sub_52 = mul_124 = None
    mul_125: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_53);  div_33 = sub_53 = None
    mul_126: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, mul_78);  mul_78 = None
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_127: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_128: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, mul_127);  mul_127 = None
    clone_19: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_128, memory_format = torch.contiguous_format);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_299: "f32[512, 256]" = torch.ops.aten.view.default(clone_19, [512, 256]);  clone_19 = None
    mm_14: "f32[512, 1024]" = torch.ops.aten.mm.default(view_299, permute_171);  permute_171 = None
    permute_172: "f32[256, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_15: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_41: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[256]" = torch.ops.aten.view.default(sum_41, [256]);  sum_41 = None
    permute_174: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_301: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_14, [1, 512, 1024]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_130: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_131: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, view_241)
    mul_132: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_131, -0.5);  mul_131 = None
    exp_17: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_132);  mul_132 = None
    mul_133: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_134: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, mul_133);  view_241 = mul_133 = None
    add_110: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_130, mul_134);  mul_130 = mul_134 = None
    mul_135: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_301, add_110);  view_301 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[512, 1024]" = torch.ops.aten.view.default(mul_135, [512, 1024]);  mul_135 = None
    mm_16: "f32[512, 256]" = torch.ops.aten.mm.default(view_302, permute_175);  permute_175 = None
    permute_176: "f32[1024, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_17: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[1024]" = torch.ops.aten.view.default(sum_42, [1024]);  sum_42 = None
    permute_178: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_304: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_16, [1, 512, 256]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_111: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_125, view_304);  mul_125 = view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_137: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, primals_176);  primals_176 = None
    mul_138: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, 256)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True)
    mul_139: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_137, mul_73);  mul_137 = None
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [2], True);  mul_139 = None
    mul_140: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, sum_44);  sum_44 = None
    sub_55: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_138, sum_43);  mul_138 = sum_43 = None
    sub_56: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_55, mul_140);  sub_55 = mul_140 = None
    mul_141: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_56);  div_34 = sub_56 = None
    mul_142: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, mul_73);  mul_73 = None
    sum_45: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1]);  mul_142 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_6: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_143: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
    mul_144: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_141, mul_143);  mul_143 = None
    clone_20: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_144, memory_format = torch.contiguous_format);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[512, 256]" = torch.ops.aten.view.default(clone_20, [512, 256]);  clone_20 = None
    mm_18: "f32[512, 256]" = torch.ops.aten.mm.default(view_305, permute_179);  permute_179 = None
    permute_180: "f32[256, 512]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_19: "f32[256, 256]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[256, 256]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_47: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[256]" = torch.ops.aten.view.default(sum_47, [256]);  sum_47 = None
    permute_182: "f32[256, 256]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_307: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_18, [1, 512, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_308: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_307, [1, 512, 4, 64]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_183, clone_default_3, clone_default_4, clone_default_5, None, alias_default_3, getitem_134, getitem_135, getitem_136, 0.1, [True, True, True, False], scale = 0.125);  permute_183 = clone_default_3 = clone_default_4 = clone_default_5 = alias_default_3 = getitem_134 = getitem_135 = getitem_136 = None
    getitem_137: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
    getitem_138: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
    getitem_139: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_189: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_315: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_22, [1, 512, 256]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_190: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    view_316: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_23, [1, 512, 256]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_317: "f32[512, 256]" = torch.ops.aten.view.default(view_316, [512, 256]);  view_316 = None
    mm_20: "f32[512, 256]" = torch.ops.aten.mm.default(view_317, permute_191);  permute_191 = None
    permute_192: "f32[256, 512]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_21: "f32[256, 256]" = torch.ops.aten.mm.default(permute_192, view_222);  permute_192 = None
    permute_193: "f32[256, 256]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_49: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_49, [256]);  sum_49 = None
    permute_194: "f32[256, 256]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_319: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_20, [1, 512, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_112: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_141, view_319);  mul_141 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_138, [0, 2, 1, 3]);  getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_320: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_195, [1, 512, 256]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_321: "f32[512, 256]" = torch.ops.aten.view.default(view_320, [512, 256]);  view_320 = None
    mm_22: "f32[512, 256]" = torch.ops.aten.mm.default(view_321, permute_196);  permute_196 = None
    permute_197: "f32[256, 512]" = torch.ops.aten.permute.default(view_321, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_197, view_222);  permute_197 = None
    permute_198: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
    view_322: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    view_323: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_22, [1, 512, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_113: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_112, view_323);  add_112 = view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_324: "f32[512, 256]" = torch.ops.aten.view.default(view_315, [512, 256]);  view_315 = None
    mm_24: "f32[512, 256]" = torch.ops.aten.mm.default(view_324, permute_200);  permute_200 = None
    permute_201: "f32[256, 512]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_201, view_222);  permute_201 = view_222 = None
    permute_202: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[256]" = torch.ops.aten.view.default(sum_51, [256]);  sum_51 = None
    permute_203: "f32[256, 256]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_326: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_24, [1, 512, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_114: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_113, view_326);  add_113 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_150: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, primals_166);  primals_166 = None
    mul_151: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_150, 256)
    sum_52: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_150, mul_71);  mul_150 = None
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, sum_53);  sum_53 = None
    sub_59: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_151, sum_52);  mul_151 = sum_52 = None
    sub_60: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_59, mul_153);  sub_59 = mul_153 = None
    mul_154: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_60);  div_36 = sub_60 = None
    mul_155: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, mul_71);  mul_71 = None
    sum_54: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_156: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_157: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, mul_156);  mul_156 = None
    clone_24: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_157, memory_format = torch.contiguous_format);  mul_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_327: "f32[512, 256]" = torch.ops.aten.view.default(clone_24, [512, 256]);  clone_24 = None
    mm_26: "f32[512, 1024]" = torch.ops.aten.mm.default(view_327, permute_204);  permute_204 = None
    permute_205: "f32[256, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_27: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_56: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[256]" = torch.ops.aten.view.default(sum_56, [256]);  sum_56 = None
    permute_207: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_329: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_26, [1, 512, 1024]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_159: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_160: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, view_219)
    mul_161: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_160, -0.5);  mul_160 = None
    exp_18: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_161);  mul_161 = None
    mul_162: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_163: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, mul_162);  view_219 = mul_162 = None
    add_116: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_159, mul_163);  mul_159 = mul_163 = None
    mul_164: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_329, add_116);  view_329 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_330: "f32[512, 1024]" = torch.ops.aten.view.default(mul_164, [512, 1024]);  mul_164 = None
    mm_28: "f32[512, 256]" = torch.ops.aten.mm.default(view_330, permute_208);  permute_208 = None
    permute_209: "f32[1024, 512]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_29: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[1024]" = torch.ops.aten.view.default(sum_57, [1024]);  sum_57 = None
    permute_211: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_332: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_28, [1, 512, 256]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_117: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_154, view_332);  mul_154 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_166: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, primals_160);  primals_160 = None
    mul_167: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_166, 256)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True)
    mul_168: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_166, mul_66);  mul_166 = None
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True);  mul_168 = None
    mul_169: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, sum_59);  sum_59 = None
    sub_62: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_167, sum_58);  mul_167 = sum_58 = None
    sub_63: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_62, mul_169);  sub_62 = mul_169 = None
    mul_170: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_63);  div_37 = sub_63 = None
    mul_171: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, mul_66);  mul_66 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1]);  mul_171 = None
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_9: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_172: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
    mul_173: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_170, mul_172);  mul_172 = None
    clone_25: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_173, memory_format = torch.contiguous_format);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[512, 256]" = torch.ops.aten.view.default(clone_25, [512, 256]);  clone_25 = None
    mm_30: "f32[512, 256]" = torch.ops.aten.mm.default(view_333, permute_212);  permute_212 = None
    permute_213: "f32[256, 512]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_31: "f32[256, 256]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[256, 256]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_62: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[256]" = torch.ops.aten.view.default(sum_62, [256]);  sum_62 = None
    permute_215: "f32[256, 256]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_335: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_30, [1, 512, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_336: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_335, [1, 512, 4, 64]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_216, clone_default_6, clone_default_7, clone_default_8, None, alias_default_5, getitem_141, getitem_142, getitem_143, 0.1, [True, True, True, False], scale = 0.125);  permute_216 = clone_default_6 = clone_default_7 = clone_default_8 = alias_default_5 = getitem_141 = getitem_142 = getitem_143 = None
    getitem_144: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
    getitem_145: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
    getitem_146: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_222: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_27: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_343: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_27, [1, 512, 256]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_223: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_28: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    view_344: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_28, [1, 512, 256]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_345: "f32[512, 256]" = torch.ops.aten.view.default(view_344, [512, 256]);  view_344 = None
    mm_32: "f32[512, 256]" = torch.ops.aten.mm.default(view_345, permute_224);  permute_224 = None
    permute_225: "f32[256, 512]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_33: "f32[256, 256]" = torch.ops.aten.mm.default(permute_225, view_200);  permute_225 = None
    permute_226: "f32[256, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_64: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[256]" = torch.ops.aten.view.default(sum_64, [256]);  sum_64 = None
    permute_227: "f32[256, 256]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_347: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_32, [1, 512, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_118: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_170, view_347);  mul_170 = view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_145, [0, 2, 1, 3]);  getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_348: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_228, [1, 512, 256]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_349: "f32[512, 256]" = torch.ops.aten.view.default(view_348, [512, 256]);  view_348 = None
    mm_34: "f32[512, 256]" = torch.ops.aten.mm.default(view_349, permute_229);  permute_229 = None
    permute_230: "f32[256, 512]" = torch.ops.aten.permute.default(view_349, [1, 0])
    mm_35: "f32[256, 256]" = torch.ops.aten.mm.default(permute_230, view_200);  permute_230 = None
    permute_231: "f32[256, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_349, [0], True);  view_349 = None
    view_350: "f32[256]" = torch.ops.aten.view.default(sum_65, [256]);  sum_65 = None
    permute_232: "f32[256, 256]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_351: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_34, [1, 512, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_119: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_118, view_351);  add_118 = view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_352: "f32[512, 256]" = torch.ops.aten.view.default(view_343, [512, 256]);  view_343 = None
    mm_36: "f32[512, 256]" = torch.ops.aten.mm.default(view_352, permute_233);  permute_233 = None
    permute_234: "f32[256, 512]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_37: "f32[256, 256]" = torch.ops.aten.mm.default(permute_234, view_200);  permute_234 = view_200 = None
    permute_235: "f32[256, 256]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[256]" = torch.ops.aten.view.default(sum_66, [256]);  sum_66 = None
    permute_236: "f32[256, 256]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_354: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_36, [1, 512, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_120: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_119, view_354);  add_119 = view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_179: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, primals_150);  primals_150 = None
    mul_180: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_179, 256)
    sum_67: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True)
    mul_181: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_179, mul_64);  mul_179 = None
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True);  mul_181 = None
    mul_182: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, sum_68);  sum_68 = None
    sub_66: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_180, sum_67);  mul_180 = sum_67 = None
    sub_67: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_66, mul_182);  sub_66 = mul_182 = None
    mul_183: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_67);  div_39 = sub_67 = None
    mul_184: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, mul_64);  mul_64 = None
    sum_69: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1]);  mul_184 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_185: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_186: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, mul_185);  mul_185 = None
    clone_29: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_186, memory_format = torch.contiguous_format);  mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_355: "f32[512, 256]" = torch.ops.aten.view.default(clone_29, [512, 256]);  clone_29 = None
    mm_38: "f32[512, 1024]" = torch.ops.aten.mm.default(view_355, permute_237);  permute_237 = None
    permute_238: "f32[256, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_39: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_71: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[256]" = torch.ops.aten.view.default(sum_71, [256]);  sum_71 = None
    permute_240: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_357: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_38, [1, 512, 1024]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_188: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_189: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_190: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_189, -0.5);  mul_189 = None
    exp_19: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_190);  mul_190 = None
    mul_191: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_192: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, mul_191);  view_197 = mul_191 = None
    add_122: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_188, mul_192);  mul_188 = mul_192 = None
    mul_193: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_357, add_122);  view_357 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[512, 1024]" = torch.ops.aten.view.default(mul_193, [512, 1024]);  mul_193 = None
    mm_40: "f32[512, 256]" = torch.ops.aten.mm.default(view_358, permute_241);  permute_241 = None
    permute_242: "f32[1024, 512]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_41: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[1024]" = torch.ops.aten.view.default(sum_72, [1024]);  sum_72 = None
    permute_244: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_360: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_40, [1, 512, 256]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_123: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_183, view_360);  mul_183 = view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_195: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, primals_144);  primals_144 = None
    mul_196: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_195, 256)
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
    mul_197: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_195, mul_59);  mul_195 = None
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    mul_198: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, sum_74);  sum_74 = None
    sub_69: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_196, sum_73);  mul_196 = sum_73 = None
    sub_70: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_69, mul_198);  sub_69 = mul_198 = None
    mul_199: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_70);  div_40 = sub_70 = None
    mul_200: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, mul_59);  mul_59 = None
    sum_75: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
    sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_12: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_201: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
    mul_202: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_199, mul_201);  mul_201 = None
    clone_30: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_202, memory_format = torch.contiguous_format);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[512, 256]" = torch.ops.aten.view.default(clone_30, [512, 256]);  clone_30 = None
    mm_42: "f32[512, 256]" = torch.ops.aten.mm.default(view_361, permute_245);  permute_245 = None
    permute_246: "f32[256, 512]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_43: "f32[256, 256]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[256, 256]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[256]" = torch.ops.aten.view.default(sum_77, [256]);  sum_77 = None
    permute_248: "f32[256, 256]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_363: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_42, [1, 512, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_364: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_363, [1, 512, 4, 64]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_249, clone_default_9, clone_default_10, clone_default_11, None, alias_default_7, getitem_148, getitem_149, getitem_150, 0.1, [True, True, True, False], scale = 0.125);  permute_249 = clone_default_9 = clone_default_10 = clone_default_11 = alias_default_7 = getitem_148 = getitem_149 = getitem_150 = None
    getitem_151: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
    getitem_152: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
    getitem_153: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_255: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_151, [0, 2, 1, 3]);  getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_32: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_371: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_32, [1, 512, 256]);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_256: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_33: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
    view_372: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_33, [1, 512, 256]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_373: "f32[512, 256]" = torch.ops.aten.view.default(view_372, [512, 256]);  view_372 = None
    mm_44: "f32[512, 256]" = torch.ops.aten.mm.default(view_373, permute_257);  permute_257 = None
    permute_258: "f32[256, 512]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_45: "f32[256, 256]" = torch.ops.aten.mm.default(permute_258, view_178);  permute_258 = None
    permute_259: "f32[256, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_79: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True);  view_373 = None
    view_374: "f32[256]" = torch.ops.aten.view.default(sum_79, [256]);  sum_79 = None
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
    view_375: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_44, [1, 512, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_124: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_199, view_375);  mul_199 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_376: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_261, [1, 512, 256]);  permute_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_377: "f32[512, 256]" = torch.ops.aten.view.default(view_376, [512, 256]);  view_376 = None
    mm_46: "f32[512, 256]" = torch.ops.aten.mm.default(view_377, permute_262);  permute_262 = None
    permute_263: "f32[256, 512]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_263, view_178);  permute_263 = None
    permute_264: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[256]" = torch.ops.aten.view.default(sum_80, [256]);  sum_80 = None
    permute_265: "f32[256, 256]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_379: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_46, [1, 512, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_125: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_124, view_379);  add_124 = view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_380: "f32[512, 256]" = torch.ops.aten.view.default(view_371, [512, 256]);  view_371 = None
    mm_48: "f32[512, 256]" = torch.ops.aten.mm.default(view_380, permute_266);  permute_266 = None
    permute_267: "f32[256, 512]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_49: "f32[256, 256]" = torch.ops.aten.mm.default(permute_267, view_178);  permute_267 = view_178 = None
    permute_268: "f32[256, 256]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[256]" = torch.ops.aten.view.default(sum_81, [256]);  sum_81 = None
    permute_269: "f32[256, 256]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_382: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_48, [1, 512, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_126: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_125, view_382);  add_125 = view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_208: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, primals_134);  primals_134 = None
    mul_209: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_208, 256)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True)
    mul_210: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_208, mul_57);  mul_208 = None
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True);  mul_210 = None
    mul_211: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, sum_83);  sum_83 = None
    sub_73: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_209, sum_82);  mul_209 = sum_82 = None
    sub_74: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_73, mul_211);  sub_73 = mul_211 = None
    mul_212: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_42, sub_74);  div_42 = sub_74 = None
    mul_213: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, mul_57);  mul_57 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_213, [0, 1]);  mul_213 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_215: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, mul_214);  mul_214 = None
    clone_34: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_215, memory_format = torch.contiguous_format);  mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_383: "f32[512, 256]" = torch.ops.aten.view.default(clone_34, [512, 256]);  clone_34 = None
    mm_50: "f32[512, 1024]" = torch.ops.aten.mm.default(view_383, permute_270);  permute_270 = None
    permute_271: "f32[256, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_51: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_86: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[256]" = torch.ops.aten.view.default(sum_86, [256]);  sum_86 = None
    permute_273: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_385: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_50, [1, 512, 1024]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_217: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_218: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, view_175)
    mul_219: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_218, -0.5);  mul_218 = None
    exp_20: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_219);  mul_219 = None
    mul_220: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_221: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_220);  view_175 = mul_220 = None
    add_128: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_217, mul_221);  mul_217 = mul_221 = None
    mul_222: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_385, add_128);  view_385 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_386: "f32[512, 1024]" = torch.ops.aten.view.default(mul_222, [512, 1024]);  mul_222 = None
    mm_52: "f32[512, 256]" = torch.ops.aten.mm.default(view_386, permute_274);  permute_274 = None
    permute_275: "f32[1024, 512]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_53: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[1024]" = torch.ops.aten.view.default(sum_87, [1024]);  sum_87 = None
    permute_277: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_388: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_52, [1, 512, 256]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_129: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_212, view_388);  mul_212 = view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_224: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, primals_128);  primals_128 = None
    mul_225: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_224, 256)
    sum_88: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_226: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_224, mul_52);  mul_224 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
    mul_227: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, sum_89);  sum_89 = None
    sub_76: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_225, sum_88);  mul_225 = sum_88 = None
    sub_77: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_76, mul_227);  sub_76 = mul_227 = None
    mul_228: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_77);  div_43 = sub_77 = None
    mul_229: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, mul_52);  mul_52 = None
    sum_90: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_15: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_230: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
    mul_231: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_228, mul_230);  mul_230 = None
    clone_35: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_231, memory_format = torch.contiguous_format);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 256]" = torch.ops.aten.view.default(clone_35, [512, 256]);  clone_35 = None
    mm_54: "f32[512, 256]" = torch.ops.aten.mm.default(view_389, permute_278);  permute_278 = None
    permute_279: "f32[256, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_55: "f32[256, 256]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[256, 256]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_92: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[256]" = torch.ops.aten.view.default(sum_92, [256]);  sum_92 = None
    permute_281: "f32[256, 256]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_391: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_54, [1, 512, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_392: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_391, [1, 512, 4, 64]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_282: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_282, clone_default_12, clone_default_13, clone_default_14, None, alias_default_9, getitem_155, getitem_156, getitem_157, 0.1, [True, True, True, False], scale = 0.125);  permute_282 = clone_default_12 = clone_default_13 = clone_default_14 = alias_default_9 = getitem_155 = getitem_156 = getitem_157 = None
    getitem_158: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
    getitem_159: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
    getitem_160: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_288: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_158, [0, 2, 1, 3]);  getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_37: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
    view_399: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_37, [1, 512, 256]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_289: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_38: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_400: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_38, [1, 512, 256]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_401: "f32[512, 256]" = torch.ops.aten.view.default(view_400, [512, 256]);  view_400 = None
    mm_56: "f32[512, 256]" = torch.ops.aten.mm.default(view_401, permute_290);  permute_290 = None
    permute_291: "f32[256, 512]" = torch.ops.aten.permute.default(view_401, [1, 0])
    mm_57: "f32[256, 256]" = torch.ops.aten.mm.default(permute_291, view_156);  permute_291 = None
    permute_292: "f32[256, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_94: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
    view_402: "f32[256]" = torch.ops.aten.view.default(sum_94, [256]);  sum_94 = None
    permute_293: "f32[256, 256]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_403: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_56, [1, 512, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_130: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_228, view_403);  mul_228 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3]);  getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_404: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_294, [1, 512, 256]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_405: "f32[512, 256]" = torch.ops.aten.view.default(view_404, [512, 256]);  view_404 = None
    mm_58: "f32[512, 256]" = torch.ops.aten.mm.default(view_405, permute_295);  permute_295 = None
    permute_296: "f32[256, 512]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_59: "f32[256, 256]" = torch.ops.aten.mm.default(permute_296, view_156);  permute_296 = None
    permute_297: "f32[256, 256]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
    view_406: "f32[256]" = torch.ops.aten.view.default(sum_95, [256]);  sum_95 = None
    permute_298: "f32[256, 256]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_407: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_58, [1, 512, 256]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_131: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_130, view_407);  add_130 = view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_408: "f32[512, 256]" = torch.ops.aten.view.default(view_399, [512, 256]);  view_399 = None
    mm_60: "f32[512, 256]" = torch.ops.aten.mm.default(view_408, permute_299);  permute_299 = None
    permute_300: "f32[256, 512]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_61: "f32[256, 256]" = torch.ops.aten.mm.default(permute_300, view_156);  permute_300 = view_156 = None
    permute_301: "f32[256, 256]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[256]" = torch.ops.aten.view.default(sum_96, [256]);  sum_96 = None
    permute_302: "f32[256, 256]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_410: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_60, [1, 512, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_132: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_131, view_410);  add_131 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_237: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, primals_118);  primals_118 = None
    mul_238: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_237, 256)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_237, mul_50);  mul_237 = None
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, sum_98);  sum_98 = None
    sub_80: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_238, sum_97);  mul_238 = sum_97 = None
    sub_81: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_80, mul_240);  sub_80 = mul_240 = None
    mul_241: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_81);  div_45 = sub_81 = None
    mul_242: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, mul_50);  mul_50 = None
    sum_99: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_243: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_244: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, mul_243);  mul_243 = None
    clone_39: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_244, memory_format = torch.contiguous_format);  mul_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_411: "f32[512, 256]" = torch.ops.aten.view.default(clone_39, [512, 256]);  clone_39 = None
    mm_62: "f32[512, 1024]" = torch.ops.aten.mm.default(view_411, permute_303);  permute_303 = None
    permute_304: "f32[256, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_63: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_101: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[256]" = torch.ops.aten.view.default(sum_101, [256]);  sum_101 = None
    permute_306: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_413: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_62, [1, 512, 1024]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_246: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_247: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, view_153)
    mul_248: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_247, -0.5);  mul_247 = None
    exp_21: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_248);  mul_248 = None
    mul_249: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_250: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, mul_249);  view_153 = mul_249 = None
    add_134: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_246, mul_250);  mul_246 = mul_250 = None
    mul_251: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_413, add_134);  view_413 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 1024]" = torch.ops.aten.view.default(mul_251, [512, 1024]);  mul_251 = None
    mm_64: "f32[512, 256]" = torch.ops.aten.mm.default(view_414, permute_307);  permute_307 = None
    permute_308: "f32[1024, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_65: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[1024]" = torch.ops.aten.view.default(sum_102, [1024]);  sum_102 = None
    permute_310: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_416: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_64, [1, 512, 256]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_135: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_241, view_416);  mul_241 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_253: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, primals_112);  primals_112 = None
    mul_254: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_253, 256)
    sum_103: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_253, mul_45);  mul_253 = None
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, sum_104);  sum_104 = None
    sub_83: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_254, sum_103);  mul_254 = sum_103 = None
    sub_84: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_83, mul_256);  sub_83 = mul_256 = None
    mul_257: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_84);  div_46 = sub_84 = None
    mul_258: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_45);  mul_45 = None
    sum_105: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_18: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_259: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
    mul_260: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_257, mul_259);  mul_259 = None
    clone_40: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_260, memory_format = torch.contiguous_format);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[512, 256]" = torch.ops.aten.view.default(clone_40, [512, 256]);  clone_40 = None
    mm_66: "f32[512, 256]" = torch.ops.aten.mm.default(view_417, permute_311);  permute_311 = None
    permute_312: "f32[256, 512]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_67: "f32[256, 256]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[256, 256]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_107: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[256]" = torch.ops.aten.view.default(sum_107, [256]);  sum_107 = None
    permute_314: "f32[256, 256]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_419: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_66, [1, 512, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_420: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_419, [1, 512, 4, 64]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_315: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_315, clone_default_15, clone_default_16, clone_default_17, None, alias_default_11, getitem_162, getitem_163, getitem_164, 0.1, [True, True, True, False], scale = 0.125);  permute_315 = clone_default_15 = clone_default_16 = clone_default_17 = alias_default_11 = getitem_162 = getitem_163 = getitem_164 = None
    getitem_165: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
    getitem_166: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
    getitem_167: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_321: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_42: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
    view_427: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_42, [1, 512, 256]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_322: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_43: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
    view_428: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_43, [1, 512, 256]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_429: "f32[512, 256]" = torch.ops.aten.view.default(view_428, [512, 256]);  view_428 = None
    mm_68: "f32[512, 256]" = torch.ops.aten.mm.default(view_429, permute_323);  permute_323 = None
    permute_324: "f32[256, 512]" = torch.ops.aten.permute.default(view_429, [1, 0])
    mm_69: "f32[256, 256]" = torch.ops.aten.mm.default(permute_324, view_134);  permute_324 = None
    permute_325: "f32[256, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_109: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True);  view_429 = None
    view_430: "f32[256]" = torch.ops.aten.view.default(sum_109, [256]);  sum_109 = None
    permute_326: "f32[256, 256]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
    view_431: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_68, [1, 512, 256]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_136: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_257, view_431);  mul_257 = view_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_166, [0, 2, 1, 3]);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_432: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_327, [1, 512, 256]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_433: "f32[512, 256]" = torch.ops.aten.view.default(view_432, [512, 256]);  view_432 = None
    mm_70: "f32[512, 256]" = torch.ops.aten.mm.default(view_433, permute_328);  permute_328 = None
    permute_329: "f32[256, 512]" = torch.ops.aten.permute.default(view_433, [1, 0])
    mm_71: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_134);  permute_329 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
    view_434: "f32[256]" = torch.ops.aten.view.default(sum_110, [256]);  sum_110 = None
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_435: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_70, [1, 512, 256]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_137: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_136, view_435);  add_136 = view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_436: "f32[512, 256]" = torch.ops.aten.view.default(view_427, [512, 256]);  view_427 = None
    mm_72: "f32[512, 256]" = torch.ops.aten.mm.default(view_436, permute_332);  permute_332 = None
    permute_333: "f32[256, 512]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_73: "f32[256, 256]" = torch.ops.aten.mm.default(permute_333, view_134);  permute_333 = view_134 = None
    permute_334: "f32[256, 256]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[256]" = torch.ops.aten.view.default(sum_111, [256]);  sum_111 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_438: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_72, [1, 512, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_138: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_137, view_438);  add_137 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_266: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, primals_102);  primals_102 = None
    mul_267: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, 256)
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True)
    mul_268: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_266, mul_43);  mul_266 = None
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True);  mul_268 = None
    mul_269: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, sum_113);  sum_113 = None
    sub_87: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_267, sum_112);  mul_267 = sum_112 = None
    sub_88: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_87, mul_269);  sub_87 = mul_269 = None
    mul_270: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_88);  div_48 = sub_88 = None
    mul_271: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, mul_43);  mul_43 = None
    sum_114: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_271, [0, 1]);  mul_271 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_272: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_273: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, mul_272);  mul_272 = None
    clone_44: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_273, memory_format = torch.contiguous_format);  mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_439: "f32[512, 256]" = torch.ops.aten.view.default(clone_44, [512, 256]);  clone_44 = None
    mm_74: "f32[512, 1024]" = torch.ops.aten.mm.default(view_439, permute_336);  permute_336 = None
    permute_337: "f32[256, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_75: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_116: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[256]" = torch.ops.aten.view.default(sum_116, [256]);  sum_116 = None
    permute_339: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_441: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_74, [1, 512, 1024]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_275: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_276: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_277: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
    exp_22: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
    mul_278: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_279: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, mul_278);  view_131 = mul_278 = None
    add_140: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
    mul_280: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_441, add_140);  view_441 = add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_442: "f32[512, 1024]" = torch.ops.aten.view.default(mul_280, [512, 1024]);  mul_280 = None
    mm_76: "f32[512, 256]" = torch.ops.aten.mm.default(view_442, permute_340);  permute_340 = None
    permute_341: "f32[1024, 512]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_77: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[1024]" = torch.ops.aten.view.default(sum_117, [1024]);  sum_117 = None
    permute_343: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_444: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_76, [1, 512, 256]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_141: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_270, view_444);  mul_270 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_282: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, primals_96);  primals_96 = None
    mul_283: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, 256)
    sum_118: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_282, mul_38);  mul_282 = None
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, sum_119);  sum_119 = None
    sub_90: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_283, sum_118);  mul_283 = sum_118 = None
    sub_91: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_90, mul_285);  sub_90 = mul_285 = None
    mul_286: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_91);  div_49 = sub_91 = None
    mul_287: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, mul_38);  mul_38 = None
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_21: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_288: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
    mul_289: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_286, mul_288);  mul_288 = None
    clone_45: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_289, memory_format = torch.contiguous_format);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[512, 256]" = torch.ops.aten.view.default(clone_45, [512, 256]);  clone_45 = None
    mm_78: "f32[512, 256]" = torch.ops.aten.mm.default(view_445, permute_344);  permute_344 = None
    permute_345: "f32[256, 512]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_79: "f32[256, 256]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[256, 256]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[256]" = torch.ops.aten.view.default(sum_122, [256]);  sum_122 = None
    permute_347: "f32[256, 256]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_447: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_78, [1, 512, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_448: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_447, [1, 512, 4, 64]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_348: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_348, clone_default_18, clone_default_19, clone_default_20, None, alias_default_13, getitem_169, getitem_170, getitem_171, 0.1, [True, True, True, False], scale = 0.125);  permute_348 = clone_default_18 = clone_default_19 = clone_default_20 = alias_default_13 = getitem_169 = getitem_170 = getitem_171 = None
    getitem_172: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
    getitem_173: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
    getitem_174: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_354: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_47: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_455: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_47, [1, 512, 256]);  clone_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_355: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_48: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
    view_456: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_48, [1, 512, 256]);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_457: "f32[512, 256]" = torch.ops.aten.view.default(view_456, [512, 256]);  view_456 = None
    mm_80: "f32[512, 256]" = torch.ops.aten.mm.default(view_457, permute_356);  permute_356 = None
    permute_357: "f32[256, 512]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_357, view_112);  permute_357 = None
    permute_358: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
    view_458: "f32[256]" = torch.ops.aten.view.default(sum_124, [256]);  sum_124 = None
    permute_359: "f32[256, 256]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
    view_459: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_80, [1, 512, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_142: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_286, view_459);  mul_286 = view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_460: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_360, [1, 512, 256]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_461: "f32[512, 256]" = torch.ops.aten.view.default(view_460, [512, 256]);  view_460 = None
    mm_82: "f32[512, 256]" = torch.ops.aten.mm.default(view_461, permute_361);  permute_361 = None
    permute_362: "f32[256, 512]" = torch.ops.aten.permute.default(view_461, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_362, view_112);  permute_362 = None
    permute_363: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
    view_462: "f32[256]" = torch.ops.aten.view.default(sum_125, [256]);  sum_125 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_463: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_82, [1, 512, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_143: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_142, view_463);  add_142 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_464: "f32[512, 256]" = torch.ops.aten.view.default(view_455, [512, 256]);  view_455 = None
    mm_84: "f32[512, 256]" = torch.ops.aten.mm.default(view_464, permute_365);  permute_365 = None
    permute_366: "f32[256, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_85: "f32[256, 256]" = torch.ops.aten.mm.default(permute_366, view_112);  permute_366 = view_112 = None
    permute_367: "f32[256, 256]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[256]" = torch.ops.aten.view.default(sum_126, [256]);  sum_126 = None
    permute_368: "f32[256, 256]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_466: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_84, [1, 512, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_144: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_143, view_466);  add_143 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_295: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, primals_86);  primals_86 = None
    mul_296: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_295, 256)
    sum_127: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
    mul_297: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_295, mul_36);  mul_295 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
    mul_298: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, sum_128);  sum_128 = None
    sub_94: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_296, sum_127);  mul_296 = sum_127 = None
    sub_95: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_94, mul_298);  sub_94 = mul_298 = None
    mul_299: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_95);  div_51 = sub_95 = None
    mul_300: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, mul_36);  mul_36 = None
    sum_129: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, mul_301);  mul_301 = None
    clone_49: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_302, memory_format = torch.contiguous_format);  mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_467: "f32[512, 256]" = torch.ops.aten.view.default(clone_49, [512, 256]);  clone_49 = None
    mm_86: "f32[512, 1024]" = torch.ops.aten.mm.default(view_467, permute_369);  permute_369 = None
    permute_370: "f32[256, 512]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_87: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[256]" = torch.ops.aten.view.default(sum_131, [256]);  sum_131 = None
    permute_372: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_469: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_86, [1, 512, 1024]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_304: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_305: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, view_109)
    mul_306: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_23: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_308: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_307);  view_109 = mul_307 = None
    add_146: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_469, add_146);  view_469 = add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_470: "f32[512, 1024]" = torch.ops.aten.view.default(mul_309, [512, 1024]);  mul_309 = None
    mm_88: "f32[512, 256]" = torch.ops.aten.mm.default(view_470, permute_373);  permute_373 = None
    permute_374: "f32[1024, 512]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_89: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[1024]" = torch.ops.aten.view.default(sum_132, [1024]);  sum_132 = None
    permute_376: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_472: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_88, [1, 512, 256]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_147: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_299, view_472);  mul_299 = view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_311: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, primals_80);  primals_80 = None
    mul_312: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_311, 256)
    sum_133: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_311, mul_31);  mul_311 = None
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, sum_134);  sum_134 = None
    sub_97: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_312, sum_133);  mul_312 = sum_133 = None
    sub_98: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_97, mul_314);  sub_97 = mul_314 = None
    mul_315: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_52, sub_98);  div_52 = sub_98 = None
    mul_316: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, mul_31);  mul_31 = None
    sum_135: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_24: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_317: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
    mul_318: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_315, mul_317);  mul_317 = None
    clone_50: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_318, memory_format = torch.contiguous_format);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[512, 256]" = torch.ops.aten.view.default(clone_50, [512, 256]);  clone_50 = None
    mm_90: "f32[512, 256]" = torch.ops.aten.mm.default(view_473, permute_377);  permute_377 = None
    permute_378: "f32[256, 512]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_91: "f32[256, 256]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[256, 256]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_137: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[256]" = torch.ops.aten.view.default(sum_137, [256]);  sum_137 = None
    permute_380: "f32[256, 256]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_475: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_90, [1, 512, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_476: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_475, [1, 512, 4, 64]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_381: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_381, clone_default_21, clone_default_22, clone_default_23, None, alias_default_15, getitem_176, getitem_177, getitem_178, 0.1, [True, True, True, False], scale = 0.125);  permute_381 = clone_default_21 = clone_default_22 = clone_default_23 = alias_default_15 = getitem_176 = getitem_177 = getitem_178 = None
    getitem_179: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
    getitem_180: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
    getitem_181: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_387: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_179, [0, 2, 1, 3]);  getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_52: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_387, memory_format = torch.contiguous_format);  permute_387 = None
    view_483: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_52, [1, 512, 256]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_388: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_53: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
    view_484: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_53, [1, 512, 256]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_485: "f32[512, 256]" = torch.ops.aten.view.default(view_484, [512, 256]);  view_484 = None
    mm_92: "f32[512, 256]" = torch.ops.aten.mm.default(view_485, permute_389);  permute_389 = None
    permute_390: "f32[256, 512]" = torch.ops.aten.permute.default(view_485, [1, 0])
    mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_390, view_90);  permute_390 = None
    permute_391: "f32[256, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
    view_486: "f32[256]" = torch.ops.aten.view.default(sum_139, [256]);  sum_139 = None
    permute_392: "f32[256, 256]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
    view_487: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_92, [1, 512, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_148: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_315, view_487);  mul_315 = view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_488: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_393, [1, 512, 256]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_489: "f32[512, 256]" = torch.ops.aten.view.default(view_488, [512, 256]);  view_488 = None
    mm_94: "f32[512, 256]" = torch.ops.aten.mm.default(view_489, permute_394);  permute_394 = None
    permute_395: "f32[256, 512]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_395, view_90);  permute_395 = None
    permute_396: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[256]" = torch.ops.aten.view.default(sum_140, [256]);  sum_140 = None
    permute_397: "f32[256, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_491: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_94, [1, 512, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_149: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_148, view_491);  add_148 = view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_492: "f32[512, 256]" = torch.ops.aten.view.default(view_483, [512, 256]);  view_483 = None
    mm_96: "f32[512, 256]" = torch.ops.aten.mm.default(view_492, permute_398);  permute_398 = None
    permute_399: "f32[256, 512]" = torch.ops.aten.permute.default(view_492, [1, 0])
    mm_97: "f32[256, 256]" = torch.ops.aten.mm.default(permute_399, view_90);  permute_399 = view_90 = None
    permute_400: "f32[256, 256]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
    view_493: "f32[256]" = torch.ops.aten.view.default(sum_141, [256]);  sum_141 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_494: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_96, [1, 512, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_150: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_149, view_494);  add_149 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_324: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, primals_70);  primals_70 = None
    mul_325: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_324, 256)
    sum_142: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
    mul_326: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_324, mul_29);  mul_324 = None
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
    mul_327: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, sum_143);  sum_143 = None
    sub_101: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_325, sum_142);  mul_325 = sum_142 = None
    sub_102: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_101, mul_327);  sub_101 = mul_327 = None
    mul_328: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_54, sub_102);  div_54 = sub_102 = None
    mul_329: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, mul_29);  mul_29 = None
    sum_144: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_330: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_331: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, mul_330);  mul_330 = None
    clone_54: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_331, memory_format = torch.contiguous_format);  mul_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_495: "f32[512, 256]" = torch.ops.aten.view.default(clone_54, [512, 256]);  clone_54 = None
    mm_98: "f32[512, 1024]" = torch.ops.aten.mm.default(view_495, permute_402);  permute_402 = None
    permute_403: "f32[256, 512]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_99: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[256]" = torch.ops.aten.view.default(sum_146, [256]);  sum_146 = None
    permute_405: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_497: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_98, [1, 512, 1024]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_333: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_334: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_335: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_24: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_337: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, mul_336);  view_87 = mul_336 = None
    add_152: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_497, add_152);  view_497 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_498: "f32[512, 1024]" = torch.ops.aten.view.default(mul_338, [512, 1024]);  mul_338 = None
    mm_100: "f32[512, 256]" = torch.ops.aten.mm.default(view_498, permute_406);  permute_406 = None
    permute_407: "f32[1024, 512]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_101: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[1024]" = torch.ops.aten.view.default(sum_147, [1024]);  sum_147 = None
    permute_409: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_500: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_100, [1, 512, 256]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_153: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_328, view_500);  mul_328 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_340: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, primals_64);  primals_64 = None
    mul_341: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_340, 256)
    sum_148: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True)
    mul_342: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_340, mul_24);  mul_340 = None
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
    mul_343: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, sum_149);  sum_149 = None
    sub_104: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_341, sum_148);  mul_341 = sum_148 = None
    sub_105: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_104, mul_343);  sub_104 = mul_343 = None
    mul_344: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_105);  div_55 = sub_105 = None
    mul_345: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, mul_24);  mul_24 = None
    sum_150: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1]);  mul_345 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_27: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_346: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
    mul_347: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_344, mul_346);  mul_346 = None
    clone_55: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_347, memory_format = torch.contiguous_format);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[512, 256]" = torch.ops.aten.view.default(clone_55, [512, 256]);  clone_55 = None
    mm_102: "f32[512, 256]" = torch.ops.aten.mm.default(view_501, permute_410);  permute_410 = None
    permute_411: "f32[256, 512]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_103: "f32[256, 256]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[256, 256]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_152: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[256]" = torch.ops.aten.view.default(sum_152, [256]);  sum_152 = None
    permute_413: "f32[256, 256]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_503: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_102, [1, 512, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_504: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_503, [1, 512, 4, 64]);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_414: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_414, clone_default_24, clone_default_25, clone_default_26, None, alias_default_17, getitem_183, getitem_184, getitem_185, 0.1, [True, True, True, False], scale = 0.125);  permute_414 = clone_default_24 = clone_default_25 = clone_default_26 = alias_default_17 = getitem_183 = getitem_184 = getitem_185 = None
    getitem_186: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
    getitem_187: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
    getitem_188: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_420: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_186, [0, 2, 1, 3]);  getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_57: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
    view_511: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_57, [1, 512, 256]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_421: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_58: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_512: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_58, [1, 512, 256]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_513: "f32[512, 256]" = torch.ops.aten.view.default(view_512, [512, 256]);  view_512 = None
    mm_104: "f32[512, 256]" = torch.ops.aten.mm.default(view_513, permute_422);  permute_422 = None
    permute_423: "f32[256, 512]" = torch.ops.aten.permute.default(view_513, [1, 0])
    mm_105: "f32[256, 256]" = torch.ops.aten.mm.default(permute_423, view_68);  permute_423 = None
    permute_424: "f32[256, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_154: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
    view_514: "f32[256]" = torch.ops.aten.view.default(sum_154, [256]);  sum_154 = None
    permute_425: "f32[256, 256]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_515: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_104, [1, 512, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_154: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_344, view_515);  mul_344 = view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_187, [0, 2, 1, 3]);  getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_516: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_426, [1, 512, 256]);  permute_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_517: "f32[512, 256]" = torch.ops.aten.view.default(view_516, [512, 256]);  view_516 = None
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(view_517, permute_427);  permute_427 = None
    permute_428: "f32[256, 512]" = torch.ops.aten.permute.default(view_517, [1, 0])
    mm_107: "f32[256, 256]" = torch.ops.aten.mm.default(permute_428, view_68);  permute_428 = None
    permute_429: "f32[256, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
    view_518: "f32[256]" = torch.ops.aten.view.default(sum_155, [256]);  sum_155 = None
    permute_430: "f32[256, 256]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_519: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_106, [1, 512, 256]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_155: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_154, view_519);  add_154 = view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_520: "f32[512, 256]" = torch.ops.aten.view.default(view_511, [512, 256]);  view_511 = None
    mm_108: "f32[512, 256]" = torch.ops.aten.mm.default(view_520, permute_431);  permute_431 = None
    permute_432: "f32[256, 512]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_432, view_68);  permute_432 = view_68 = None
    permute_433: "f32[256, 256]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
    view_521: "f32[256]" = torch.ops.aten.view.default(sum_156, [256]);  sum_156 = None
    permute_434: "f32[256, 256]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_522: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_108, [1, 512, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_156: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_155, view_522);  add_155 = view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_353: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, primals_54);  primals_54 = None
    mul_354: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_353, 256)
    sum_157: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_353, mul_22);  mul_353 = None
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_158);  sum_158 = None
    sub_108: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_354, sum_157);  mul_354 = sum_157 = None
    sub_109: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_108, mul_356);  sub_108 = mul_356 = None
    mul_357: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_109);  div_57 = sub_109 = None
    mul_358: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, mul_22);  mul_22 = None
    sum_159: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_360: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, mul_359);  mul_359 = None
    clone_59: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_360, memory_format = torch.contiguous_format);  mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_523: "f32[512, 256]" = torch.ops.aten.view.default(clone_59, [512, 256]);  clone_59 = None
    mm_110: "f32[512, 1024]" = torch.ops.aten.mm.default(view_523, permute_435);  permute_435 = None
    permute_436: "f32[256, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_111: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_161: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[256]" = torch.ops.aten.view.default(sum_161, [256]);  sum_161 = None
    permute_438: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_525: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_110, [1, 512, 1024]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_362: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_363: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, view_65)
    mul_364: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_363, -0.5);  mul_363 = None
    exp_25: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_364);  mul_364 = None
    mul_365: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_366: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, mul_365);  view_65 = mul_365 = None
    add_158: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_362, mul_366);  mul_362 = mul_366 = None
    mul_367: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_525, add_158);  view_525 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[512, 1024]" = torch.ops.aten.view.default(mul_367, [512, 1024]);  mul_367 = None
    mm_112: "f32[512, 256]" = torch.ops.aten.mm.default(view_526, permute_439);  permute_439 = None
    permute_440: "f32[1024, 512]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_113: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[1024]" = torch.ops.aten.view.default(sum_162, [1024]);  sum_162 = None
    permute_442: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_528: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_112, [1, 512, 256]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_159: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_357, view_528);  mul_357 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_369: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, primals_48);  primals_48 = None
    mul_370: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_369, 256)
    sum_163: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True)
    mul_371: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_369, mul_17);  mul_369 = None
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True);  mul_371 = None
    mul_372: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, sum_164);  sum_164 = None
    sub_111: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_370, sum_163);  mul_370 = sum_163 = None
    sub_112: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_111, mul_372);  sub_111 = mul_372 = None
    mul_373: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_58, sub_112);  div_58 = sub_112 = None
    mul_374: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, mul_17);  mul_17 = None
    sum_165: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1]);  mul_374 = None
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_30: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_375: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
    mul_376: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_373, mul_375);  mul_375 = None
    clone_60: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_376, memory_format = torch.contiguous_format);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[512, 256]" = torch.ops.aten.view.default(clone_60, [512, 256]);  clone_60 = None
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(view_529, permute_443);  permute_443 = None
    permute_444: "f32[256, 512]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_115: "f32[256, 256]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[256, 256]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_167: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[256]" = torch.ops.aten.view.default(sum_167, [256]);  sum_167 = None
    permute_446: "f32[256, 256]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_531: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_114, [1, 512, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_532: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_531, [1, 512, 4, 64]);  view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_447: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_447, clone_default_27, clone_default_28, clone_default_29, None, alias_default_19, getitem_190, getitem_191, getitem_192, 0.1, [True, True, True, False], scale = 0.125);  permute_447 = clone_default_27 = clone_default_28 = clone_default_29 = alias_default_19 = getitem_190 = getitem_191 = getitem_192 = None
    getitem_193: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
    getitem_194: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
    getitem_195: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_453: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_193, [0, 2, 1, 3]);  getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_62: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
    view_539: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_62, [1, 512, 256]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_454: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_63: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
    view_540: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_63, [1, 512, 256]);  clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_541: "f32[512, 256]" = torch.ops.aten.view.default(view_540, [512, 256]);  view_540 = None
    mm_116: "f32[512, 256]" = torch.ops.aten.mm.default(view_541, permute_455);  permute_455 = None
    permute_456: "f32[256, 512]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_117: "f32[256, 256]" = torch.ops.aten.mm.default(permute_456, view_46);  permute_456 = None
    permute_457: "f32[256, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_169: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
    view_542: "f32[256]" = torch.ops.aten.view.default(sum_169, [256]);  sum_169 = None
    permute_458: "f32[256, 256]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_543: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_116, [1, 512, 256]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_160: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_373, view_543);  mul_373 = view_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_194, [0, 2, 1, 3]);  getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_544: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_459, [1, 512, 256]);  permute_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_545: "f32[512, 256]" = torch.ops.aten.view.default(view_544, [512, 256]);  view_544 = None
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(view_545, permute_460);  permute_460 = None
    permute_461: "f32[256, 512]" = torch.ops.aten.permute.default(view_545, [1, 0])
    mm_119: "f32[256, 256]" = torch.ops.aten.mm.default(permute_461, view_46);  permute_461 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
    view_546: "f32[256]" = torch.ops.aten.view.default(sum_170, [256]);  sum_170 = None
    permute_463: "f32[256, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_547: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_118, [1, 512, 256]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_161: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_160, view_547);  add_160 = view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_548: "f32[512, 256]" = torch.ops.aten.view.default(view_539, [512, 256]);  view_539 = None
    mm_120: "f32[512, 256]" = torch.ops.aten.mm.default(view_548, permute_464);  permute_464 = None
    permute_465: "f32[256, 512]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_121: "f32[256, 256]" = torch.ops.aten.mm.default(permute_465, view_46);  permute_465 = view_46 = None
    permute_466: "f32[256, 256]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_467: "f32[256, 256]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_550: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_120, [1, 512, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_162: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_161, view_550);  add_161 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_382: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, primals_38);  primals_38 = None
    mul_383: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_382, 256)
    sum_172: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_382, [2], True)
    mul_384: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_382, mul_15);  mul_382 = None
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [2], True);  mul_384 = None
    mul_385: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, sum_173);  sum_173 = None
    sub_115: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_383, sum_172);  mul_383 = sum_172 = None
    sub_116: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_115, mul_385);  sub_115 = mul_385 = None
    mul_386: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_60, sub_116);  div_60 = sub_116 = None
    mul_387: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, mul_15);  mul_15 = None
    sum_174: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 1]);  mul_387 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_388: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_389: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, mul_388);  mul_388 = None
    clone_64: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_389, memory_format = torch.contiguous_format);  mul_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_551: "f32[512, 256]" = torch.ops.aten.view.default(clone_64, [512, 256]);  clone_64 = None
    mm_122: "f32[512, 1024]" = torch.ops.aten.mm.default(view_551, permute_468);  permute_468 = None
    permute_469: "f32[256, 512]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_123: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_176: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[256]" = torch.ops.aten.view.default(sum_176, [256]);  sum_176 = None
    permute_471: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_553: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_122, [1, 512, 1024]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_391: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_392: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_393: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_392, -0.5);  mul_392 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_393);  mul_393 = None
    mul_394: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_395: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_394);  view_43 = mul_394 = None
    add_164: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_391, mul_395);  mul_391 = mul_395 = None
    mul_396: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_553, add_164);  view_553 = add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_554: "f32[512, 1024]" = torch.ops.aten.view.default(mul_396, [512, 1024]);  mul_396 = None
    mm_124: "f32[512, 256]" = torch.ops.aten.mm.default(view_554, permute_472);  permute_472 = None
    permute_473: "f32[1024, 512]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_125: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[1024]" = torch.ops.aten.view.default(sum_177, [1024]);  sum_177 = None
    permute_475: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_556: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_124, [1, 512, 256]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_165: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_386, view_556);  mul_386 = view_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_398: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, primals_32);  primals_32 = None
    mul_399: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_398, 256)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True)
    mul_400: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_398, mul_10);  mul_398 = None
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [2], True);  mul_400 = None
    mul_401: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, sum_179);  sum_179 = None
    sub_118: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_399, sum_178);  mul_399 = sum_178 = None
    sub_119: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_118, mul_401);  sub_118 = mul_401 = None
    mul_402: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_119);  div_61 = sub_119 = None
    mul_403: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, mul_10);  mul_10 = None
    sum_180: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1]);  mul_403 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_33: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_404: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
    mul_405: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_402, mul_404);  mul_404 = None
    clone_65: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_405, memory_format = torch.contiguous_format);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[512, 256]" = torch.ops.aten.view.default(clone_65, [512, 256]);  clone_65 = None
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(view_557, permute_476);  permute_476 = None
    permute_477: "f32[256, 512]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_127: "f32[256, 256]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[256, 256]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_182: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[256]" = torch.ops.aten.view.default(sum_182, [256]);  sum_182 = None
    permute_479: "f32[256, 256]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_559: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_126, [1, 512, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_560: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_559, [1, 512, 4, 64]);  view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_480: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_480, clone_default_30, clone_default_31, clone_default_32, None, alias_default_21, getitem_197, getitem_198, getitem_199, 0.1, [True, True, True, False], scale = 0.125);  permute_480 = clone_default_30 = clone_default_31 = clone_default_32 = alias_default_21 = getitem_197 = getitem_198 = getitem_199 = None
    getitem_200: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
    getitem_201: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
    getitem_202: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_486: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_200, [0, 2, 1, 3]);  getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_67: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_486, memory_format = torch.contiguous_format);  permute_486 = None
    view_567: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_67, [1, 512, 256]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_487: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_68: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_568: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_68, [1, 512, 256]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_569: "f32[512, 256]" = torch.ops.aten.view.default(view_568, [512, 256]);  view_568 = None
    mm_128: "f32[512, 256]" = torch.ops.aten.mm.default(view_569, permute_488);  permute_488 = None
    permute_489: "f32[256, 512]" = torch.ops.aten.permute.default(view_569, [1, 0])
    mm_129: "f32[256, 256]" = torch.ops.aten.mm.default(permute_489, view_24);  permute_489 = None
    permute_490: "f32[256, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_184: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
    view_570: "f32[256]" = torch.ops.aten.view.default(sum_184, [256]);  sum_184 = None
    permute_491: "f32[256, 256]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_571: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_128, [1, 512, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_166: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_402, view_571);  mul_402 = view_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_201, [0, 2, 1, 3]);  getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_572: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_492, [1, 512, 256]);  permute_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_573: "f32[512, 256]" = torch.ops.aten.view.default(view_572, [512, 256]);  view_572 = None
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(view_573, permute_493);  permute_493 = None
    permute_494: "f32[256, 512]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_131: "f32[256, 256]" = torch.ops.aten.mm.default(permute_494, view_24);  permute_494 = None
    permute_495: "f32[256, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
    view_574: "f32[256]" = torch.ops.aten.view.default(sum_185, [256]);  sum_185 = None
    permute_496: "f32[256, 256]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_575: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_130, [1, 512, 256]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_167: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_166, view_575);  add_166 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_576: "f32[512, 256]" = torch.ops.aten.view.default(view_567, [512, 256]);  view_567 = None
    mm_132: "f32[512, 256]" = torch.ops.aten.mm.default(view_576, permute_497);  permute_497 = None
    permute_498: "f32[256, 512]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_133: "f32[256, 256]" = torch.ops.aten.mm.default(permute_498, view_24);  permute_498 = view_24 = None
    permute_499: "f32[256, 256]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[256]" = torch.ops.aten.view.default(sum_186, [256]);  sum_186 = None
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_578: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_132, [1, 512, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_168: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_167, view_578);  add_167 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_411: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, primals_22);  primals_22 = None
    mul_412: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_411, 256)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
    mul_413: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_411, mul_8);  mul_411 = None
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    mul_414: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, sum_188);  sum_188 = None
    sub_122: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_412, sum_187);  mul_412 = sum_187 = None
    sub_123: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_122, mul_414);  sub_122 = mul_414 = None
    mul_415: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_63, sub_123);  div_63 = sub_123 = None
    mul_416: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, mul_8);  mul_8 = None
    sum_189: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
    sum_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_417: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_418: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, mul_417);  mul_417 = None
    clone_69: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_579: "f32[512, 256]" = torch.ops.aten.view.default(clone_69, [512, 256]);  clone_69 = None
    mm_134: "f32[512, 1024]" = torch.ops.aten.mm.default(view_579, permute_501);  permute_501 = None
    permute_502: "f32[256, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_135: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_191: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[256]" = torch.ops.aten.view.default(sum_191, [256]);  sum_191 = None
    permute_504: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_581: "f32[1, 512, 1024]" = torch.ops.aten.view.default(mm_134, [1, 512, 1024]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_420: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_421: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_422: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_421, -0.5);  mul_421 = None
    exp_27: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_422);  mul_422 = None
    mul_423: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_424: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_423);  view_21 = mul_423 = None
    add_170: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_420, mul_424);  mul_420 = mul_424 = None
    mul_425: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_581, add_170);  view_581 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_582: "f32[512, 1024]" = torch.ops.aten.view.default(mul_425, [512, 1024]);  mul_425 = None
    mm_136: "f32[512, 256]" = torch.ops.aten.mm.default(view_582, permute_505);  permute_505 = None
    permute_506: "f32[1024, 512]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_137: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[1024]" = torch.ops.aten.view.default(sum_192, [1024]);  sum_192 = None
    permute_508: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_584: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_136, [1, 512, 256]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_171: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_415, view_584);  mul_415 = view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_427: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, primals_16);  primals_16 = None
    mul_428: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_427, 256)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
    mul_429: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_427, mul_3);  mul_427 = None
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
    mul_430: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, sum_194);  sum_194 = None
    sub_125: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_428, sum_193);  mul_428 = sum_193 = None
    sub_126: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_125, mul_430);  sub_125 = mul_430 = None
    mul_431: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_64, sub_126);  div_64 = sub_126 = None
    mul_432: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, mul_3);  mul_3 = None
    sum_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_36: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_433: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
    mul_434: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_431, mul_433);  mul_433 = None
    clone_70: "f32[1, 512, 256]" = torch.ops.aten.clone.default(mul_434, memory_format = torch.contiguous_format);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[512, 256]" = torch.ops.aten.view.default(clone_70, [512, 256]);  clone_70 = None
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(view_585, permute_509);  permute_509 = None
    permute_510: "f32[256, 512]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_139: "f32[256, 256]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[256, 256]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_197: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[256]" = torch.ops.aten.view.default(sum_197, [256]);  sum_197 = None
    permute_512: "f32[256, 256]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_587: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_138, [1, 512, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_588: "f32[1, 512, 4, 64]" = torch.ops.aten.view.default(view_587, [1, 512, 4, 64]);  view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_513: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # No stacktrace found for following nodes
    _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_513, clone_default_33, clone_default_34, clone_default_35, None, alias_default_23, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale = 0.125);  permute_513 = clone_default_33 = clone_default_34 = clone_default_35 = alias_default_23 = getitem_204 = getitem_205 = getitem_206 = None
    getitem_207: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
    getitem_208: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
    getitem_209: "f32[1, 4, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_519: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_72: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_519, memory_format = torch.contiguous_format);  permute_519 = None
    view_595: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_72, [1, 512, 256]);  clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_520: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_73: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
    view_596: "f32[1, 512, 256]" = torch.ops.aten.view.default(clone_73, [1, 512, 256]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_597: "f32[512, 256]" = torch.ops.aten.view.default(view_596, [512, 256]);  view_596 = None
    mm_140: "f32[512, 256]" = torch.ops.aten.mm.default(view_597, permute_521);  permute_521 = None
    permute_522: "f32[256, 512]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_141: "f32[256, 256]" = torch.ops.aten.mm.default(permute_522, view_2);  permute_522 = None
    permute_523: "f32[256, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[256]" = torch.ops.aten.view.default(sum_199, [256]);  sum_199 = None
    permute_524: "f32[256, 256]" = torch.ops.aten.permute.default(permute_523, [1, 0]);  permute_523 = None
    view_599: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_140, [1, 512, 256]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_172: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_431, view_599);  mul_431 = view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(getitem_208, [0, 2, 1, 3]);  getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_600: "f32[1, 512, 256]" = torch.ops.aten.view.default(permute_525, [1, 512, 256]);  permute_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_601: "f32[512, 256]" = torch.ops.aten.view.default(view_600, [512, 256]);  view_600 = None
    mm_142: "f32[512, 256]" = torch.ops.aten.mm.default(view_601, permute_526);  permute_526 = None
    permute_527: "f32[256, 512]" = torch.ops.aten.permute.default(view_601, [1, 0])
    mm_143: "f32[256, 256]" = torch.ops.aten.mm.default(permute_527, view_2);  permute_527 = None
    permute_528: "f32[256, 256]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_601, [0], True);  view_601 = None
    view_602: "f32[256]" = torch.ops.aten.view.default(sum_200, [256]);  sum_200 = None
    permute_529: "f32[256, 256]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_603: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_142, [1, 512, 256]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_173: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_172, view_603);  add_172 = view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_604: "f32[512, 256]" = torch.ops.aten.view.default(view_595, [512, 256]);  view_595 = None
    mm_144: "f32[512, 256]" = torch.ops.aten.mm.default(view_604, permute_530);  permute_530 = None
    permute_531: "f32[256, 512]" = torch.ops.aten.permute.default(view_604, [1, 0])
    mm_145: "f32[256, 256]" = torch.ops.aten.mm.default(permute_531, view_2);  permute_531 = view_2 = None
    permute_532: "f32[256, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
    view_605: "f32[256]" = torch.ops.aten.view.default(sum_201, [256]);  sum_201 = None
    permute_533: "f32[256, 256]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_606: "f32[1, 512, 256]" = torch.ops.aten.view.default(mm_144, [1, 512, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_174: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_173, view_606);  add_173 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view_607: "f32[512, 256]" = torch.ops.aten.view.default(add_174, [512, 256]);  add_174 = None
    mm_146: "f32[512, 128]" = torch.ops.aten.mm.default(view_607, permute_534);  permute_534 = None
    permute_535: "f32[256, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_147: "f32[256, 128]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[128, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[256]" = torch.ops.aten.view.default(sum_202, [256]);  sum_202 = None
    permute_537: "f32[256, 128]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_609: "f32[1, 512, 128]" = torch.ops.aten.view.default(mm_146, [1, 512, 128]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    convert_element_type_38: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_439: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
    mul_440: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_609, mul_439);  view_609 = mul_439 = None
    clone_74: "f32[1, 512, 128]" = torch.ops.aten.clone.default(mul_440, memory_format = torch.contiguous_format);  mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    mul_442: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_74, primals_4);  primals_4 = None
    mul_443: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_442, 128)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True)
    mul_444: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_442, mul_1);  mul_442 = None
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True);  mul_444 = None
    mul_445: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, sum_204);  sum_204 = None
    sub_129: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_443, sum_203);  mul_443 = sum_203 = None
    sub_130: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_129, mul_445);  sub_129 = mul_445 = None
    mul_446: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_66, sub_130);  div_66 = sub_130 = None
    mul_447: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(clone_74, mul_1);  mul_1 = None
    sum_205: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 1]);  mul_447 = None
    sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_74, [0, 1]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_8: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_8: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_8, full_default_2, mul_446);  unsqueeze_8 = None
    full_default_12: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_12, [slice_4], where_8, True);  full_default_12 = slice_4 = where_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_9: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_9: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_9, full_default_2, mul_446);  unsqueeze_9 = None
    full_default_14: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_14, [expand], where_9, True);  full_default_14 = expand = where_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_204, 0)
    unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_10: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, mul_446);  unsqueeze_10 = full_default_2 = mul_446 = None
    full_default_16: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 128]" = torch.ops.aten._unsafe_index_put.default(full_default_16, [primals_204], where_10, True);  full_default_16 = primals_204 = where_10 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_205, sum_206, permute_537, view_608, permute_533, view_605, permute_529, view_602, permute_524, view_598, permute_512, view_586, sum_195, sum_196, permute_508, view_583, permute_504, view_580, sum_189, sum_190, permute_500, view_577, permute_496, view_574, permute_491, view_570, permute_479, view_558, sum_180, sum_181, permute_475, view_555, permute_471, view_552, sum_174, sum_175, permute_467, view_549, permute_463, view_546, permute_458, view_542, permute_446, view_530, sum_165, sum_166, permute_442, view_527, permute_438, view_524, sum_159, sum_160, permute_434, view_521, permute_430, view_518, permute_425, view_514, permute_413, view_502, sum_150, sum_151, permute_409, view_499, permute_405, view_496, sum_144, sum_145, permute_401, view_493, permute_397, view_490, permute_392, view_486, permute_380, view_474, sum_135, sum_136, permute_376, view_471, permute_372, view_468, sum_129, sum_130, permute_368, view_465, permute_364, view_462, permute_359, view_458, permute_347, view_446, sum_120, sum_121, permute_343, view_443, permute_339, view_440, sum_114, sum_115, permute_335, view_437, permute_331, view_434, permute_326, view_430, permute_314, view_418, sum_105, sum_106, permute_310, view_415, permute_306, view_412, sum_99, sum_100, permute_302, view_409, permute_298, view_406, permute_293, view_402, permute_281, view_390, sum_90, sum_91, permute_277, view_387, permute_273, view_384, sum_84, sum_85, permute_269, view_381, permute_265, view_378, permute_260, view_374, permute_248, view_362, sum_75, sum_76, permute_244, view_359, permute_240, view_356, sum_69, sum_70, permute_236, view_353, permute_232, view_350, permute_227, view_346, permute_215, view_334, sum_60, sum_61, permute_211, view_331, permute_207, view_328, sum_54, sum_55, permute_203, view_325, permute_199, view_322, permute_194, view_318, permute_182, view_306, sum_45, sum_46, permute_178, view_303, permute_174, view_300, sum_39, sum_40, permute_170, view_297, permute_166, view_294, permute_161, view_290, permute_149, view_278, sum_30, sum_31, permute_145, view_275, permute_141, view_272, sum_24, sum_25, permute_137, view_269, None, None, None, None, None]
    