from __future__ import annotations



def forward(self, primals_4: "f32[768]", primals_14: "f32[768]", primals_20: "f32[768]", primals_30: "f32[768]", primals_36: "f32[768]", primals_46: "f32[768]", primals_52: "f32[768]", primals_62: "f32[768]", primals_68: "f32[768]", primals_78: "f32[768]", primals_84: "f32[768]", primals_94: "f32[768]", primals_100: "f32[768]", primals_110: "f32[768]", primals_116: "f32[768]", primals_126: "f32[768]", primals_132: "f32[768]", primals_142: "f32[768]", primals_148: "f32[768]", primals_158: "f32[768]", primals_164: "f32[768]", primals_174: "f32[768]", primals_180: "f32[768]", primals_190: "f32[768]", primals_196: "f32[768]", primals_200: "f32[768]", primals_206: "i64[4, 512]", expand: "i64[4, 512]", slice_4: "i64[1, 512]", mul_1: "f32[4, 512, 768]", view: "f32[2048, 768]", view_16: "f32[2048, 768]", mul_3: "f32[4, 512, 768]", view_18: "f32[2048, 768]", addmm_4: "f32[2048, 3072]", view_20: "f32[2048, 3072]", mul_8: "f32[4, 512, 768]", view_22: "f32[2048, 768]", view_38: "f32[2048, 768]", mul_10: "f32[4, 512, 768]", view_40: "f32[2048, 768]", addmm_10: "f32[2048, 3072]", view_42: "f32[2048, 3072]", mul_15: "f32[4, 512, 768]", view_44: "f32[2048, 768]", view_60: "f32[2048, 768]", mul_17: "f32[4, 512, 768]", view_62: "f32[2048, 768]", addmm_16: "f32[2048, 3072]", view_64: "f32[2048, 3072]", mul_22: "f32[4, 512, 768]", view_66: "f32[2048, 768]", view_82: "f32[2048, 768]", mul_24: "f32[4, 512, 768]", view_84: "f32[2048, 768]", addmm_22: "f32[2048, 3072]", view_86: "f32[2048, 3072]", mul_29: "f32[4, 512, 768]", view_88: "f32[2048, 768]", view_104: "f32[2048, 768]", mul_31: "f32[4, 512, 768]", view_106: "f32[2048, 768]", addmm_28: "f32[2048, 3072]", view_108: "f32[2048, 3072]", mul_36: "f32[4, 512, 768]", view_110: "f32[2048, 768]", view_126: "f32[2048, 768]", mul_38: "f32[4, 512, 768]", view_128: "f32[2048, 768]", addmm_34: "f32[2048, 3072]", view_130: "f32[2048, 3072]", mul_43: "f32[4, 512, 768]", view_132: "f32[2048, 768]", view_148: "f32[2048, 768]", mul_45: "f32[4, 512, 768]", view_150: "f32[2048, 768]", addmm_40: "f32[2048, 3072]", view_152: "f32[2048, 3072]", mul_50: "f32[4, 512, 768]", view_154: "f32[2048, 768]", view_170: "f32[2048, 768]", mul_52: "f32[4, 512, 768]", view_172: "f32[2048, 768]", addmm_46: "f32[2048, 3072]", view_174: "f32[2048, 3072]", mul_57: "f32[4, 512, 768]", view_176: "f32[2048, 768]", view_192: "f32[2048, 768]", mul_59: "f32[4, 512, 768]", view_194: "f32[2048, 768]", addmm_52: "f32[2048, 3072]", view_196: "f32[2048, 3072]", mul_64: "f32[4, 512, 768]", view_198: "f32[2048, 768]", view_214: "f32[2048, 768]", mul_66: "f32[4, 512, 768]", view_216: "f32[2048, 768]", addmm_58: "f32[2048, 3072]", view_218: "f32[2048, 3072]", mul_71: "f32[4, 512, 768]", view_220: "f32[2048, 768]", view_236: "f32[2048, 768]", mul_73: "f32[4, 512, 768]", view_238: "f32[2048, 768]", addmm_64: "f32[2048, 3072]", view_240: "f32[2048, 3072]", mul_78: "f32[4, 512, 768]", view_242: "f32[2048, 768]", view_258: "f32[2048, 768]", mul_80: "f32[4, 512, 768]", view_260: "f32[2048, 768]", addmm_70: "f32[2048, 3072]", view_262: "f32[2048, 3072]", mul_85: "f32[4, 512, 768]", view_264: "f32[2048, 768]", addmm_72: "f32[2048, 768]", mul_90: "f32[4, 512, 768]", view_266: "f32[2048, 768]", permute_134: "f32[30522, 768]", div_24: "f32[4, 512, 1]", permute_138: "f32[768, 768]", div_25: "f32[4, 512, 1]", permute_142: "f32[768, 3072]", permute_146: "f32[3072, 768]", div_26: "f32[4, 512, 1]", permute_150: "f32[768, 768]", permute_155: "f32[48, 512, 512]", permute_156: "f32[48, 64, 512]", alias_12: "f32[4, 12, 512, 512]", permute_157: "f32[48, 64, 512]", permute_158: "f32[48, 512, 64]", permute_162: "f32[768, 768]", permute_167: "f32[768, 768]", permute_171: "f32[768, 768]", div_28: "f32[4, 512, 1]", permute_175: "f32[768, 3072]", permute_179: "f32[3072, 768]", div_29: "f32[4, 512, 1]", permute_183: "f32[768, 768]", permute_188: "f32[48, 512, 512]", permute_189: "f32[48, 64, 512]", alias_13: "f32[4, 12, 512, 512]", permute_190: "f32[48, 64, 512]", permute_191: "f32[48, 512, 64]", permute_195: "f32[768, 768]", permute_200: "f32[768, 768]", permute_204: "f32[768, 768]", div_31: "f32[4, 512, 1]", permute_208: "f32[768, 3072]", permute_212: "f32[3072, 768]", div_32: "f32[4, 512, 1]", permute_216: "f32[768, 768]", permute_221: "f32[48, 512, 512]", permute_222: "f32[48, 64, 512]", alias_14: "f32[4, 12, 512, 512]", permute_223: "f32[48, 64, 512]", permute_224: "f32[48, 512, 64]", permute_228: "f32[768, 768]", permute_233: "f32[768, 768]", permute_237: "f32[768, 768]", div_34: "f32[4, 512, 1]", permute_241: "f32[768, 3072]", permute_245: "f32[3072, 768]", div_35: "f32[4, 512, 1]", permute_249: "f32[768, 768]", permute_254: "f32[48, 512, 512]", permute_255: "f32[48, 64, 512]", alias_15: "f32[4, 12, 512, 512]", permute_256: "f32[48, 64, 512]", permute_257: "f32[48, 512, 64]", permute_261: "f32[768, 768]", permute_266: "f32[768, 768]", permute_270: "f32[768, 768]", div_37: "f32[4, 512, 1]", permute_274: "f32[768, 3072]", permute_278: "f32[3072, 768]", div_38: "f32[4, 512, 1]", permute_282: "f32[768, 768]", permute_287: "f32[48, 512, 512]", permute_288: "f32[48, 64, 512]", alias_16: "f32[4, 12, 512, 512]", permute_289: "f32[48, 64, 512]", permute_290: "f32[48, 512, 64]", permute_294: "f32[768, 768]", permute_299: "f32[768, 768]", permute_303: "f32[768, 768]", div_40: "f32[4, 512, 1]", permute_307: "f32[768, 3072]", permute_311: "f32[3072, 768]", div_41: "f32[4, 512, 1]", permute_315: "f32[768, 768]", permute_320: "f32[48, 512, 512]", permute_321: "f32[48, 64, 512]", alias_17: "f32[4, 12, 512, 512]", permute_322: "f32[48, 64, 512]", permute_323: "f32[48, 512, 64]", permute_327: "f32[768, 768]", permute_332: "f32[768, 768]", permute_336: "f32[768, 768]", div_43: "f32[4, 512, 1]", permute_340: "f32[768, 3072]", permute_344: "f32[3072, 768]", div_44: "f32[4, 512, 1]", permute_348: "f32[768, 768]", permute_353: "f32[48, 512, 512]", permute_354: "f32[48, 64, 512]", alias_18: "f32[4, 12, 512, 512]", permute_355: "f32[48, 64, 512]", permute_356: "f32[48, 512, 64]", permute_360: "f32[768, 768]", permute_365: "f32[768, 768]", permute_369: "f32[768, 768]", div_46: "f32[4, 512, 1]", permute_373: "f32[768, 3072]", permute_377: "f32[3072, 768]", div_47: "f32[4, 512, 1]", permute_381: "f32[768, 768]", permute_386: "f32[48, 512, 512]", permute_387: "f32[48, 64, 512]", alias_19: "f32[4, 12, 512, 512]", permute_388: "f32[48, 64, 512]", permute_389: "f32[48, 512, 64]", permute_393: "f32[768, 768]", permute_398: "f32[768, 768]", permute_402: "f32[768, 768]", div_49: "f32[4, 512, 1]", permute_406: "f32[768, 3072]", permute_410: "f32[3072, 768]", div_50: "f32[4, 512, 1]", permute_414: "f32[768, 768]", permute_419: "f32[48, 512, 512]", permute_420: "f32[48, 64, 512]", alias_20: "f32[4, 12, 512, 512]", permute_421: "f32[48, 64, 512]", permute_422: "f32[48, 512, 64]", permute_426: "f32[768, 768]", permute_431: "f32[768, 768]", permute_435: "f32[768, 768]", div_52: "f32[4, 512, 1]", permute_439: "f32[768, 3072]", permute_443: "f32[3072, 768]", div_53: "f32[4, 512, 1]", permute_447: "f32[768, 768]", permute_452: "f32[48, 512, 512]", permute_453: "f32[48, 64, 512]", alias_21: "f32[4, 12, 512, 512]", permute_454: "f32[48, 64, 512]", permute_455: "f32[48, 512, 64]", permute_459: "f32[768, 768]", permute_464: "f32[768, 768]", permute_468: "f32[768, 768]", div_55: "f32[4, 512, 1]", permute_472: "f32[768, 3072]", permute_476: "f32[3072, 768]", div_56: "f32[4, 512, 1]", permute_480: "f32[768, 768]", permute_485: "f32[48, 512, 512]", permute_486: "f32[48, 64, 512]", alias_22: "f32[4, 12, 512, 512]", permute_487: "f32[48, 64, 512]", permute_488: "f32[48, 512, 64]", permute_492: "f32[768, 768]", permute_497: "f32[768, 768]", permute_501: "f32[768, 768]", div_58: "f32[4, 512, 1]", permute_505: "f32[768, 3072]", permute_509: "f32[3072, 768]", div_59: "f32[4, 512, 1]", permute_513: "f32[768, 768]", permute_518: "f32[48, 512, 512]", permute_519: "f32[48, 64, 512]", alias_23: "f32[4, 12, 512, 512]", permute_520: "f32[48, 64, 512]", permute_521: "f32[48, 512, 64]", permute_525: "f32[768, 768]", permute_530: "f32[768, 768]", permute_534: "f32[768, 768]", div_61: "f32[4, 512, 1]", tangents_1: "f32[4, 512, 30522]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_19: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_4, [4, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_41: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_63: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_16, [4, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_85: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_107: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_28, [4, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_129: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_151: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_40, [4, 512, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_173: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_52, [4, 512, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_64, [4, 512, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_261: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[4, 512, 3072]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    view_265: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(addmm_72, [4, 512, 768]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, 0.7071067811865476)
    erf_12: "f32[4, 512, 768]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:702, code: hidden_states = self.decoder(hidden_states)
    view_268: "f32[2048, 30522]" = torch.ops.aten.reshape.default(tangents_1, [2048, 30522]);  tangents_1 = None
    mm: "f32[2048, 768]" = torch.ops.aten.mm.default(view_268, permute_134);  permute_134 = None
    permute_135: "f32[30522, 2048]" = torch.ops.aten.permute.default(view_268, [1, 0])
    mm_1: "f32[30522, 768]" = torch.ops.aten.mm.default(permute_135, view_266);  permute_135 = view_266 = None
    permute_136: "f32[768, 30522]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_268, [0], True);  view_268 = None
    view_269: "f32[30522]" = torch.ops.aten.reshape.default(sum_13, [30522]);  sum_13 = None
    permute_137: "f32[30522, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_270: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm, [4, 512, 768]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:682, code: hidden_states = self.LayerNorm(hidden_states)
    mul_93: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_270, primals_200);  primals_200 = None
    mul_94: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, 768)
    sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_93, [2], True)
    mul_95: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_93, mul_90);  mul_93 = None
    sum_15: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True);  mul_95 = None
    mul_96: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_15);  sum_15 = None
    sub_40: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_94, sum_14);  mul_94 = sum_14 = None
    sub_41: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_40, mul_96);  sub_40 = mul_96 = None
    mul_97: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_41);  div_24 = sub_41 = None
    mul_98: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_270, mul_90);  mul_90 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1]);  mul_98 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_270, [0, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_100: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_100, 0.5);  add_100 = None
    mul_101: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, view_265)
    mul_102: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_101, -0.5);  mul_101 = None
    exp_12: "f32[4, 512, 768]" = torch.ops.aten.exp.default(mul_102);  mul_102 = None
    mul_103: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_104: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_103);  view_265 = mul_103 = None
    add_104: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_100, mul_104);  mul_100 = mul_104 = None
    mul_105: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_97, add_104);  mul_97 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:680, code: hidden_states = self.dense(hidden_states)
    view_271: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_105, [2048, 768]);  mul_105 = None
    mm_2: "f32[2048, 768]" = torch.ops.aten.mm.default(view_271, permute_138);  permute_138 = None
    permute_139: "f32[768, 2048]" = torch.ops.aten.permute.default(view_271, [1, 0])
    mm_3: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_264);  permute_139 = view_264 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_271, [0], True);  view_271 = None
    view_272: "f32[768]" = torch.ops.aten.reshape.default(sum_18, [768]);  sum_18 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_273: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_107: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, primals_196);  primals_196 = None
    mul_108: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, 768)
    sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_107, [2], True)
    mul_109: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_107, mul_85);  mul_107 = None
    sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_20);  sum_20 = None
    sub_43: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_108, sum_19);  mul_108 = sum_19 = None
    sub_44: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_43, mul_110);  sub_43 = mul_110 = None
    mul_111: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_44);  div_25 = sub_44 = None
    mul_112: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(view_273, mul_85);  mul_85 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_112, [0, 1]);  mul_112 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_273, [0, 1]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_274: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_111, [2048, 768])
    mm_4: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_274, permute_142);  permute_142 = None
    permute_143: "f32[768, 2048]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_143, view_262);  permute_143 = view_262 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
    permute_145: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    view_276: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_4, [4, 512, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_114: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_115: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_116: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_115, -0.5);  mul_115 = None
    exp_13: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_116);  mul_116 = None
    mul_117: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_118: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_117);  view_261 = mul_117 = None
    add_106: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_114, mul_118);  mul_114 = mul_118 = None
    mul_119: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_276, add_106);  view_276 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_277: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_119, [2048, 3072]);  mul_119 = None
    mm_6: "f32[2048, 768]" = torch.ops.aten.mm.default(view_277, permute_146);  permute_146 = None
    permute_147: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_147, view_260);  permute_147 = view_260 = None
    permute_148: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_24: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_277, [0], True);  view_277 = None
    view_278: "f32[3072]" = torch.ops.aten.reshape.default(sum_24, [3072]);  sum_24 = None
    permute_149: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_279: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_6, [4, 512, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_107: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_111, view_279);  mul_111 = view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_121: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, primals_190);  primals_190 = None
    mul_122: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, 768)
    sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True)
    mul_123: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_121, mul_80);  mul_121 = None
    sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True);  mul_123 = None
    mul_124: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_26);  sum_26 = None
    sub_46: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_122, sum_25);  mul_122 = sum_25 = None
    sub_47: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_46, mul_124);  sub_46 = mul_124 = None
    mul_125: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_47);  div_26 = sub_47 = None
    mul_126: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_107, mul_80);  mul_80 = None
    sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_126, [0, 1]);  mul_126 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_107, [0, 1]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_280: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_125, [2048, 768])
    mm_8: "f32[2048, 768]" = torch.ops.aten.mm.default(view_280, permute_150);  permute_150 = None
    permute_151: "f32[768, 2048]" = torch.ops.aten.permute.default(view_280, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_151, view_258);  permute_151 = view_258 = None
    permute_152: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_280, [0], True);  view_280 = None
    view_281: "f32[768]" = torch.ops.aten.reshape.default(sum_29, [768]);  sum_29 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_282: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_8, [4, 512, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_283: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_282, [4, 512, 12, 64]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_154: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_283, [0, 2, 1, 3]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_85: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
    view_284: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_85, [48, 512, 64]);  clone_85 = None
    bmm_24: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_155, view_284);  permute_155 = None
    bmm_25: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_284, permute_156);  view_284 = permute_156 = None
    view_285: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_24, [4, 12, 512, 64]);  bmm_24 = None
    view_286: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_25, [4, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_127: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_286, alias_12);  view_286 = None
    sum_30: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
    mul_128: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_12, sum_30);  alias_12 = sum_30 = None
    sub_48: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_127, mul_128);  mul_127 = mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_27: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_48, 8.0);  sub_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_287: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_27, [48, 512, 512]);  div_27 = None
    bmm_26: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_157, view_287);  permute_157 = None
    bmm_27: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_287, permute_158);  view_287 = permute_158 = None
    view_288: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_26, [4, 12, 64, 512]);  bmm_26 = None
    view_289: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_27, [4, 12, 512, 64]);  bmm_27 = None
    permute_159: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 1, 3, 2]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_160: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_86: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_290: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_86, [4, 512, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_87: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_291: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_87, [4, 512, 768]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_292: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_291, [2048, 768]);  view_291 = None
    mm_10: "f32[2048, 768]" = torch.ops.aten.mm.default(view_292, permute_162);  permute_162 = None
    permute_163: "f32[768, 2048]" = torch.ops.aten.permute.default(view_292, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True);  view_292 = None
    view_293: "f32[768]" = torch.ops.aten.reshape.default(sum_31, [768]);  sum_31 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
    view_294: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_10, [4, 512, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_108: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_125, view_294);  mul_125 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_166: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_159, [0, 2, 1, 3]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_295: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_166, [4, 512, 768]);  permute_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_88: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_295, memory_format = torch.contiguous_format);  view_295 = None
    view_296: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_88, [2048, 768]);  clone_88 = None
    mm_12: "f32[2048, 768]" = torch.ops.aten.mm.default(view_296, permute_167);  permute_167 = None
    permute_168: "f32[768, 2048]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_13: "f32[768, 768]" = torch.ops.aten.mm.default(permute_168, view_242);  permute_168 = None
    permute_169: "f32[768, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[768]" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
    permute_170: "f32[768, 768]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    view_298: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_12, [4, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_109: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_108, view_298);  add_108 = view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_299: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_290, [2048, 768]);  view_290 = None
    mm_14: "f32[2048, 768]" = torch.ops.aten.mm.default(view_299, permute_171);  permute_171 = None
    permute_172: "f32[768, 2048]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_172, view_242);  permute_172 = view_242 = None
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[768]" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    view_301: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_14, [4, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_110: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_109, view_301);  add_109 = view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_130: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, primals_180);  primals_180 = None
    mul_131: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, 768)
    sum_34: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
    mul_132: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_78);  mul_130 = None
    sum_35: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
    mul_133: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_35);  sum_35 = None
    sub_50: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_131, sum_34);  mul_131 = sum_34 = None
    sub_51: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_133);  sub_50 = mul_133 = None
    mul_134: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_51);  div_28 = sub_51 = None
    mul_135: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_110, mul_78);  mul_78 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_110, [0, 1]);  add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_302: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_134, [2048, 768])
    mm_16: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_302, permute_175);  permute_175 = None
    permute_176: "f32[768, 2048]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_176, view_240);  permute_176 = view_240 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
    view_303: "f32[768]" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
    permute_178: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    view_304: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_16, [4, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_137: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_138: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_139: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_138, -0.5);  mul_138 = None
    exp_14: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_139);  mul_139 = None
    mul_140: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_141: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_140);  view_239 = mul_140 = None
    add_112: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_137, mul_141);  mul_137 = mul_141 = None
    mul_142: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_304, add_112);  view_304 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_305: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_142, [2048, 3072]);  mul_142 = None
    mm_18: "f32[2048, 768]" = torch.ops.aten.mm.default(view_305, permute_179);  permute_179 = None
    permute_180: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_305, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_180, view_238);  permute_180 = view_238 = None
    permute_181: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
    view_306: "f32[3072]" = torch.ops.aten.reshape.default(sum_39, [3072]);  sum_39 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_307: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_18, [4, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_113: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_134, view_307);  mul_134 = view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_144: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_174);  primals_174 = None
    mul_145: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_144, 768)
    sum_40: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True)
    mul_146: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_144, mul_73);  mul_144 = None
    sum_41: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_146, [2], True);  mul_146 = None
    mul_147: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_41);  sum_41 = None
    sub_53: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_145, sum_40);  mul_145 = sum_40 = None
    sub_54: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_147);  sub_53 = mul_147 = None
    mul_148: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_54);  div_29 = sub_54 = None
    mul_149: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_113, mul_73);  mul_73 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 1]);  mul_149 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1]);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_308: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_148, [2048, 768])
    mm_20: "f32[2048, 768]" = torch.ops.aten.mm.default(view_308, permute_183);  permute_183 = None
    permute_184: "f32[768, 2048]" = torch.ops.aten.permute.default(view_308, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_184, view_236);  permute_184 = view_236 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_308, [0], True);  view_308 = None
    view_309: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_310: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_20, [4, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_311: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_310, [4, 512, 12, 64]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_187: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_311, [0, 2, 1, 3]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_89: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
    view_312: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_89, [48, 512, 64]);  clone_89 = None
    bmm_28: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_188, view_312);  permute_188 = None
    bmm_29: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_312, permute_189);  view_312 = permute_189 = None
    view_313: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_28, [4, 12, 512, 64]);  bmm_28 = None
    view_314: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_29, [4, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_150: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_314, alias_13);  view_314 = None
    sum_45: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [-1], True)
    mul_151: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_13, sum_45);  alias_13 = sum_45 = None
    sub_55: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_30: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_55, 8.0);  sub_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_315: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_30, [48, 512, 512]);  div_30 = None
    bmm_30: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_190, view_315);  permute_190 = None
    bmm_31: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_315, permute_191);  view_315 = permute_191 = None
    view_316: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_30, [4, 12, 64, 512]);  bmm_30 = None
    view_317: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_31, [4, 12, 512, 64]);  bmm_31 = None
    permute_192: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 1, 3, 2]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_193: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_90: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_318: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_90, [4, 512, 768]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_91: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_319: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_91, [4, 512, 768]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_320: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_319, [2048, 768]);  view_319 = None
    mm_22: "f32[2048, 768]" = torch.ops.aten.mm.default(view_320, permute_195);  permute_195 = None
    permute_196: "f32[768, 2048]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[768]" = torch.ops.aten.reshape.default(sum_46, [768]);  sum_46 = None
    permute_198: "f32[768, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_322: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_22, [4, 512, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_114: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_148, view_322);  mul_148 = view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_323: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_199, [4, 512, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_92: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_323, memory_format = torch.contiguous_format);  view_323 = None
    view_324: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_92, [2048, 768]);  clone_92 = None
    mm_24: "f32[2048, 768]" = torch.ops.aten.mm.default(view_324, permute_200);  permute_200 = None
    permute_201: "f32[768, 2048]" = torch.ops.aten.permute.default(view_324, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_201, view_220);  permute_201 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_324, [0], True);  view_324 = None
    view_325: "f32[768]" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
    permute_203: "f32[768, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_326: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_24, [4, 512, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_115: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_114, view_326);  add_114 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_327: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_318, [2048, 768]);  view_318 = None
    mm_26: "f32[2048, 768]" = torch.ops.aten.mm.default(view_327, permute_204);  permute_204 = None
    permute_205: "f32[768, 2048]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_27: "f32[768, 768]" = torch.ops.aten.mm.default(permute_205, view_220);  permute_205 = view_220 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_327, [0], True);  view_327 = None
    view_328: "f32[768]" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
    permute_207: "f32[768, 768]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_329: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_26, [4, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_116: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_115, view_329);  add_115 = view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_153: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_164);  primals_164 = None
    mul_154: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, 768)
    sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True)
    mul_155: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_153, mul_71);  mul_153 = None
    sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True);  mul_155 = None
    mul_156: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_71, sum_50);  sum_50 = None
    sub_57: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_154, sum_49);  mul_154 = sum_49 = None
    sub_58: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_57, mul_156);  sub_57 = mul_156 = None
    mul_157: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_58);  div_31 = sub_58 = None
    mul_158: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_116, mul_71);  mul_71 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1]);  mul_158 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_116, [0, 1]);  add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_330: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_157, [2048, 768])
    mm_28: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_330, permute_208);  permute_208 = None
    permute_209: "f32[768, 2048]" = torch.ops.aten.permute.default(view_330, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_218);  permute_209 = view_218 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_330, [0], True);  view_330 = None
    view_331: "f32[768]" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_332: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_28, [4, 512, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_160: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_161: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_162: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_161, -0.5);  mul_161 = None
    exp_15: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_162);  mul_162 = None
    mul_163: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_164: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_163);  view_217 = mul_163 = None
    add_118: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_160, mul_164);  mul_160 = mul_164 = None
    mul_165: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_332, add_118);  view_332 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_333: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_165, [2048, 3072]);  mul_165 = None
    mm_30: "f32[2048, 768]" = torch.ops.aten.mm.default(view_333, permute_212);  permute_212 = None
    permute_213: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_333, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_216);  permute_213 = view_216 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_54: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_333, [0], True);  view_333 = None
    view_334: "f32[3072]" = torch.ops.aten.reshape.default(sum_54, [3072]);  sum_54 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_335: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_30, [4, 512, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_119: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_157, view_335);  mul_157 = view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_167: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, primals_158);  primals_158 = None
    mul_168: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
    sum_55: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_66);  mul_167 = None
    sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_66, sum_56);  sum_56 = None
    sub_60: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_55);  mul_168 = sum_55 = None
    sub_61: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_60, mul_170);  sub_60 = mul_170 = None
    mul_171: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_61);  div_32 = sub_61 = None
    mul_172: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_119, mul_66);  mul_66 = None
    sum_57: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_336: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_171, [2048, 768])
    mm_32: "f32[2048, 768]" = torch.ops.aten.mm.default(view_336, permute_216);  permute_216 = None
    permute_217: "f32[768, 2048]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_214);  permute_217 = view_214 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
    view_337: "f32[768]" = torch.ops.aten.reshape.default(sum_59, [768]);  sum_59 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_338: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_32, [4, 512, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_339: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_338, [4, 512, 12, 64]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_220: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_93: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    view_340: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_93, [48, 512, 64]);  clone_93 = None
    bmm_32: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_221, view_340);  permute_221 = None
    bmm_33: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_340, permute_222);  view_340 = permute_222 = None
    view_341: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_32, [4, 12, 512, 64]);  bmm_32 = None
    view_342: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_33, [4, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_173: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_342, alias_14);  view_342 = None
    sum_60: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [-1], True)
    mul_174: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_14, sum_60);  alias_14 = sum_60 = None
    sub_62: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_173, mul_174);  mul_173 = mul_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_33: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_62, 8.0);  sub_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_343: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_33, [48, 512, 512]);  div_33 = None
    bmm_34: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_223, view_343);  permute_223 = None
    bmm_35: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_343, permute_224);  view_343 = permute_224 = None
    view_344: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_34, [4, 12, 64, 512]);  bmm_34 = None
    view_345: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_35, [4, 12, 512, 64]);  bmm_35 = None
    permute_225: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_344, [0, 1, 3, 2]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_226: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_94: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_346: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_94, [4, 512, 768]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_95: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_347: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_95, [4, 512, 768]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_348: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_347, [2048, 768]);  view_347 = None
    mm_34: "f32[2048, 768]" = torch.ops.aten.mm.default(view_348, permute_228);  permute_228 = None
    permute_229: "f32[768, 2048]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True);  view_348 = None
    view_349: "f32[768]" = torch.ops.aten.reshape.default(sum_61, [768]);  sum_61 = None
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
    view_350: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_34, [4, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_120: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_171, view_350);  mul_171 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_232: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_225, [0, 2, 1, 3]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_351: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_232, [4, 512, 768]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_96: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_351, memory_format = torch.contiguous_format);  view_351 = None
    view_352: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_96, [2048, 768]);  clone_96 = None
    mm_36: "f32[2048, 768]" = torch.ops.aten.mm.default(view_352, permute_233);  permute_233 = None
    permute_234: "f32[768, 2048]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_234, view_198);  permute_234 = None
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_352, [0], True);  view_352 = None
    view_353: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
    permute_236: "f32[768, 768]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    view_354: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_36, [4, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_121: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_120, view_354);  add_120 = view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_355: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_346, [2048, 768]);  view_346 = None
    mm_38: "f32[2048, 768]" = torch.ops.aten.mm.default(view_355, permute_237);  permute_237 = None
    permute_238: "f32[768, 2048]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_238, view_198);  permute_238 = view_198 = None
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_63: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
    view_357: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_38, [4, 512, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_122: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_121, view_357);  add_121 = view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_176: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_148);  primals_148 = None
    mul_177: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, 768)
    sum_64: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True)
    mul_178: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_176, mul_64);  mul_176 = None
    sum_65: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True);  mul_178 = None
    mul_179: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_64, sum_65);  sum_65 = None
    sub_64: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_177, sum_64);  mul_177 = sum_64 = None
    sub_65: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_64, mul_179);  sub_64 = mul_179 = None
    mul_180: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_65);  div_34 = sub_65 = None
    mul_181: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, mul_64);  mul_64 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_181, [0, 1]);  mul_181 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1]);  add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_358: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_180, [2048, 768])
    mm_40: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_358, permute_241);  permute_241 = None
    permute_242: "f32[768, 2048]" = torch.ops.aten.permute.default(view_358, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_242, view_196);  permute_242 = view_196 = None
    permute_243: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_358, [0], True);  view_358 = None
    view_359: "f32[768]" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
    permute_244: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_360: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_40, [4, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_183: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_184: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_185: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_184, -0.5);  mul_184 = None
    exp_16: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_185);  mul_185 = None
    mul_186: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_187: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_186);  view_195 = mul_186 = None
    add_124: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_183, mul_187);  mul_183 = mul_187 = None
    mul_188: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_360, add_124);  view_360 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_361: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_188, [2048, 3072]);  mul_188 = None
    mm_42: "f32[2048, 768]" = torch.ops.aten.mm.default(view_361, permute_245);  permute_245 = None
    permute_246: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_361, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_246, view_194);  permute_246 = view_194 = None
    permute_247: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_69: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
    view_362: "f32[3072]" = torch.ops.aten.reshape.default(sum_69, [3072]);  sum_69 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_363: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_42, [4, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_125: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_180, view_363);  mul_180 = view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_190: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_142);  primals_142 = None
    mul_191: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_190, 768)
    sum_70: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True)
    mul_192: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_190, mul_59);  mul_190 = None
    sum_71: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
    mul_193: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_59, sum_71);  sum_71 = None
    sub_67: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_191, sum_70);  mul_191 = sum_70 = None
    sub_68: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_67, mul_193);  sub_67 = mul_193 = None
    mul_194: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_68);  div_35 = sub_68 = None
    mul_195: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_125, mul_59);  mul_59 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
    sum_73: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_194, [2048, 768])
    mm_44: "f32[2048, 768]" = torch.ops.aten.mm.default(view_364, permute_249);  permute_249 = None
    permute_250: "f32[768, 2048]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_250, view_192);  permute_250 = view_192 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[768]" = torch.ops.aten.reshape.default(sum_74, [768]);  sum_74 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_366: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_44, [4, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_366, [4, 512, 12, 64]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_253: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_97: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    view_368: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_97, [48, 512, 64]);  clone_97 = None
    bmm_36: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_254, view_368);  permute_254 = None
    bmm_37: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_368, permute_255);  view_368 = permute_255 = None
    view_369: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_36, [4, 12, 512, 64]);  bmm_36 = None
    view_370: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_37, [4, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_196: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_370, alias_15);  view_370 = None
    sum_75: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [-1], True)
    mul_197: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_15, sum_75);  alias_15 = sum_75 = None
    sub_69: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_36: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_69, 8.0);  sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_371: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_36, [48, 512, 512]);  div_36 = None
    bmm_38: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_256, view_371);  permute_256 = None
    bmm_39: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_371, permute_257);  view_371 = permute_257 = None
    view_372: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_38, [4, 12, 64, 512]);  bmm_38 = None
    view_373: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_39, [4, 12, 512, 64]);  bmm_39 = None
    permute_258: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 1, 3, 2]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_259: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_98: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_374: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_98, [4, 512, 768]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_99: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_375: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_99, [4, 512, 768]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_376: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_375, [2048, 768]);  view_375 = None
    mm_46: "f32[2048, 768]" = torch.ops.aten.mm.default(view_376, permute_261);  permute_261 = None
    permute_262: "f32[768, 2048]" = torch.ops.aten.permute.default(view_376, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_376, [0], True);  view_376 = None
    view_377: "f32[768]" = torch.ops.aten.reshape.default(sum_76, [768]);  sum_76 = None
    permute_264: "f32[768, 768]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_378: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_46, [4, 512, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_126: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_194, view_378);  mul_194 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_265: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_258, [0, 2, 1, 3]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_379: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_265, [4, 512, 768]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_100: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_379, memory_format = torch.contiguous_format);  view_379 = None
    view_380: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_100, [2048, 768]);  clone_100 = None
    mm_48: "f32[2048, 768]" = torch.ops.aten.mm.default(view_380, permute_266);  permute_266 = None
    permute_267: "f32[768, 2048]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_176);  permute_267 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[768]" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_382: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_48, [4, 512, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_127: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_126, view_382);  add_126 = view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_383: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_374, [2048, 768]);  view_374 = None
    mm_50: "f32[2048, 768]" = torch.ops.aten.mm.default(view_383, permute_270);  permute_270 = None
    permute_271: "f32[768, 2048]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_51: "f32[768, 768]" = torch.ops.aten.mm.default(permute_271, view_176);  permute_271 = view_176 = None
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[768]" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_385: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_50, [4, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_128: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_127, view_385);  add_127 = view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_199: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_132);  primals_132 = None
    mul_200: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, 768)
    sum_79: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True)
    mul_201: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_199, mul_57);  mul_199 = None
    sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
    mul_202: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_80);  sum_80 = None
    sub_71: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_200, sum_79);  mul_200 = sum_79 = None
    sub_72: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_202);  sub_71 = mul_202 = None
    mul_203: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_72);  div_37 = sub_72 = None
    mul_204: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_128, mul_57);  mul_57 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_204, [0, 1]);  mul_204 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_128, [0, 1]);  add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_386: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_203, [2048, 768])
    mm_52: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_386, permute_274);  permute_274 = None
    permute_275: "f32[768, 2048]" = torch.ops.aten.permute.default(view_386, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_275, view_174);  permute_275 = view_174 = None
    permute_276: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
    view_387: "f32[768]" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
    permute_277: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_388: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_52, [4, 512, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_206: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_207: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_208: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
    exp_17: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
    mul_209: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_210: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_209);  view_173 = mul_209 = None
    add_130: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
    mul_211: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_388, add_130);  view_388 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_211, [2048, 3072]);  mul_211 = None
    mm_54: "f32[2048, 768]" = torch.ops.aten.mm.default(view_389, permute_278);  permute_278 = None
    permute_279: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_279, view_172);  permute_279 = view_172 = None
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[3072]" = torch.ops.aten.reshape.default(sum_84, [3072]);  sum_84 = None
    permute_281: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_391: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_54, [4, 512, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_131: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_203, view_391);  mul_203 = view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_213: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_126);  primals_126 = None
    mul_214: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_85: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_52);  mul_213 = None
    sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_52, sum_86);  sum_86 = None
    sub_74: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_85);  mul_214 = sum_85 = None
    sub_75: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_216);  sub_74 = mul_216 = None
    mul_217: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_38, sub_75);  div_38 = sub_75 = None
    mul_218: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_52);  mul_52 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_392: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_217, [2048, 768])
    mm_56: "f32[2048, 768]" = torch.ops.aten.mm.default(view_392, permute_282);  permute_282 = None
    permute_283: "f32[768, 2048]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_283, view_170);  permute_283 = view_170 = None
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[768]" = torch.ops.aten.reshape.default(sum_89, [768]);  sum_89 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(permute_284, [1, 0]);  permute_284 = None
    view_394: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_56, [4, 512, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_395: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_394, [4, 512, 12, 64]);  view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_286: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_101: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None
    view_396: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_101, [48, 512, 64]);  clone_101 = None
    bmm_40: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_287, view_396);  permute_287 = None
    bmm_41: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_396, permute_288);  view_396 = permute_288 = None
    view_397: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_40, [4, 12, 512, 64]);  bmm_40 = None
    view_398: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_41, [4, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_219: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_398, alias_16);  view_398 = None
    sum_90: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_219, [-1], True)
    mul_220: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_16, sum_90);  alias_16 = sum_90 = None
    sub_76: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_219, mul_220);  mul_219 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_39: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_76, 8.0);  sub_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_399: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_39, [48, 512, 512]);  div_39 = None
    bmm_42: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_289, view_399);  permute_289 = None
    bmm_43: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_399, permute_290);  view_399 = permute_290 = None
    view_400: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_42, [4, 12, 64, 512]);  bmm_42 = None
    view_401: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_43, [4, 12, 512, 64]);  bmm_43 = None
    permute_291: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 1, 3, 2]);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_292: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_401, [0, 2, 1, 3]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_102: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_402: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_102, [4, 512, 768]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_103: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_403: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_103, [4, 512, 768]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_404: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_403, [2048, 768]);  view_403 = None
    mm_58: "f32[2048, 768]" = torch.ops.aten.mm.default(view_404, permute_294);  permute_294 = None
    permute_295: "f32[768, 2048]" = torch.ops.aten.permute.default(view_404, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_404, [0], True);  view_404 = None
    view_405: "f32[768]" = torch.ops.aten.reshape.default(sum_91, [768]);  sum_91 = None
    permute_297: "f32[768, 768]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_406: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_58, [4, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_132: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_217, view_406);  mul_217 = view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_291, [0, 2, 1, 3]);  permute_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_407: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_298, [4, 512, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_104: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_407, memory_format = torch.contiguous_format);  view_407 = None
    view_408: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_104, [2048, 768]);  clone_104 = None
    mm_60: "f32[2048, 768]" = torch.ops.aten.mm.default(view_408, permute_299);  permute_299 = None
    permute_300: "f32[768, 2048]" = torch.ops.aten.permute.default(view_408, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_300, view_154);  permute_300 = None
    permute_301: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_408, [0], True);  view_408 = None
    view_409: "f32[768]" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    view_410: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_60, [4, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_133: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_132, view_410);  add_132 = view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_411: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_402, [2048, 768]);  view_402 = None
    mm_62: "f32[2048, 768]" = torch.ops.aten.mm.default(view_411, permute_303);  permute_303 = None
    permute_304: "f32[768, 2048]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_304, view_154);  permute_304 = view_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_93: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
    permute_306: "f32[768, 768]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_413: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_62, [4, 512, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_134: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_133, view_413);  add_133 = view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_222: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_116);  primals_116 = None
    mul_223: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
    sum_94: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_50);  mul_222 = None
    sum_95: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_95);  sum_95 = None
    sub_78: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_94);  mul_223 = sum_94 = None
    sub_79: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_225);  sub_78 = mul_225 = None
    mul_226: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_40, sub_79);  div_40 = sub_79 = None
    mul_227: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_50);  mul_50 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_226, [2048, 768])
    mm_64: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_414, permute_307);  permute_307 = None
    permute_308: "f32[768, 2048]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_308, view_152);  permute_308 = view_152 = None
    permute_309: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.reshape.default(sum_98, [768]);  sum_98 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_309, [1, 0]);  permute_309 = None
    view_416: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_64, [4, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_229: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_230: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_231: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_230, -0.5);  mul_230 = None
    exp_18: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_231);  mul_231 = None
    mul_232: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_233: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_232);  view_151 = mul_232 = None
    add_136: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_229, mul_233);  mul_229 = mul_233 = None
    mul_234: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_416, add_136);  view_416 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_417: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_234, [2048, 3072]);  mul_234 = None
    mm_66: "f32[2048, 768]" = torch.ops.aten.mm.default(view_417, permute_311);  permute_311 = None
    permute_312: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_417, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_312, view_150);  permute_312 = view_150 = None
    permute_313: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_99: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
    view_418: "f32[3072]" = torch.ops.aten.reshape.default(sum_99, [3072]);  sum_99 = None
    permute_314: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
    view_419: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_66, [4, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_137: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_226, view_419);  mul_226 = view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_236: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_110);  primals_110 = None
    mul_237: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, 768)
    sum_100: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True)
    mul_238: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_236, mul_45);  mul_236 = None
    sum_101: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True);  mul_238 = None
    mul_239: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_101);  sum_101 = None
    sub_81: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_237, sum_100);  mul_237 = sum_100 = None
    sub_82: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_81, mul_239);  sub_81 = mul_239 = None
    mul_240: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_82);  div_41 = sub_82 = None
    mul_241: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_137, mul_45);  mul_45 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 1]);  mul_241 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_137, [0, 1]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_420: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_240, [2048, 768])
    mm_68: "f32[2048, 768]" = torch.ops.aten.mm.default(view_420, permute_315);  permute_315 = None
    permute_316: "f32[768, 2048]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_316, view_148);  permute_316 = view_148 = None
    permute_317: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_104: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[768]" = torch.ops.aten.reshape.default(sum_104, [768]);  sum_104 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
    view_422: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_68, [4, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_423: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_422, [4, 512, 12, 64]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_319: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_423, [0, 2, 1, 3]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_105: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_424: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_105, [48, 512, 64]);  clone_105 = None
    bmm_44: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_320, view_424);  permute_320 = None
    bmm_45: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_424, permute_321);  view_424 = permute_321 = None
    view_425: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_44, [4, 12, 512, 64]);  bmm_44 = None
    view_426: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_45, [4, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_242: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_426, alias_17);  view_426 = None
    sum_105: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [-1], True)
    mul_243: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_17, sum_105);  alias_17 = sum_105 = None
    sub_83: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_42: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_83, 8.0);  sub_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_427: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_42, [48, 512, 512]);  div_42 = None
    bmm_46: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_322, view_427);  permute_322 = None
    bmm_47: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_427, permute_323);  view_427 = permute_323 = None
    view_428: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_46, [4, 12, 64, 512]);  bmm_46 = None
    view_429: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_47, [4, 12, 512, 64]);  bmm_47 = None
    permute_324: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 1, 3, 2]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_429, [0, 2, 1, 3]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_106: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_430: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_106, [4, 512, 768]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_107: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_431: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_107, [4, 512, 768]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_432: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_431, [2048, 768]);  view_431 = None
    mm_70: "f32[2048, 768]" = torch.ops.aten.mm.default(view_432, permute_327);  permute_327 = None
    permute_328: "f32[768, 2048]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_432, [0], True);  view_432 = None
    view_433: "f32[768]" = torch.ops.aten.reshape.default(sum_106, [768]);  sum_106 = None
    permute_330: "f32[768, 768]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    view_434: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_70, [4, 512, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_138: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_240, view_434);  mul_240 = view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_331: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_324, [0, 2, 1, 3]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_435: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_331, [4, 512, 768]);  permute_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_108: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_435, memory_format = torch.contiguous_format);  view_435 = None
    view_436: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_108, [2048, 768]);  clone_108 = None
    mm_72: "f32[2048, 768]" = torch.ops.aten.mm.default(view_436, permute_332);  permute_332 = None
    permute_333: "f32[768, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_333, view_132);  permute_333 = None
    permute_334: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[768]" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
    permute_335: "f32[768, 768]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_438: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_72, [4, 512, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_139: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_138, view_438);  add_138 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_439: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_430, [2048, 768]);  view_430 = None
    mm_74: "f32[2048, 768]" = torch.ops.aten.mm.default(view_439, permute_336);  permute_336 = None
    permute_337: "f32[768, 2048]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_75: "f32[768, 768]" = torch.ops.aten.mm.default(permute_337, view_132);  permute_337 = view_132 = None
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
    permute_339: "f32[768, 768]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_441: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_74, [4, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_140: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_441);  add_139 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_245: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_100);  primals_100 = None
    mul_246: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_245, 768)
    sum_109: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
    mul_247: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_245, mul_43);  mul_245 = None
    sum_110: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
    mul_248: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_43, sum_110);  sum_110 = None
    sub_85: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_246, sum_109);  mul_246 = sum_109 = None
    sub_86: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_248);  sub_85 = mul_248 = None
    mul_249: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_43, sub_86);  div_43 = sub_86 = None
    mul_250: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_140, mul_43);  mul_43 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_442: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_249, [2048, 768])
    mm_76: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_442, permute_340);  permute_340 = None
    permute_341: "f32[768, 2048]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_341, view_130);  permute_341 = view_130 = None
    permute_342: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
    permute_343: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_444: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_76, [4, 512, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_252: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_253: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_254: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_253, -0.5);  mul_253 = None
    exp_19: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_254);  mul_254 = None
    mul_255: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_256: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_255);  view_129 = mul_255 = None
    add_142: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_252, mul_256);  mul_252 = mul_256 = None
    mul_257: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_444, add_142);  view_444 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_445: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_257, [2048, 3072]);  mul_257 = None
    mm_78: "f32[2048, 768]" = torch.ops.aten.mm.default(view_445, permute_344);  permute_344 = None
    permute_345: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_345, view_128);  permute_345 = view_128 = None
    permute_346: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[3072]" = torch.ops.aten.reshape.default(sum_114, [3072]);  sum_114 = None
    permute_347: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_447: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_78, [4, 512, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_143: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_249, view_447);  mul_249 = view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_259: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, primals_94);  primals_94 = None
    mul_260: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_259, 768)
    sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True)
    mul_261: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_259, mul_38);  mul_259 = None
    sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_261, [2], True);  mul_261 = None
    mul_262: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_116);  sum_116 = None
    sub_88: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_260, sum_115);  mul_260 = sum_115 = None
    sub_89: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_88, mul_262);  sub_88 = mul_262 = None
    mul_263: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_44, sub_89);  div_44 = sub_89 = None
    mul_264: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_143, mul_38);  mul_38 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 1]);  mul_264 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_448: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_263, [2048, 768])
    mm_80: "f32[2048, 768]" = torch.ops.aten.mm.default(view_448, permute_348);  permute_348 = None
    permute_349: "f32[768, 2048]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_349, view_126);  permute_349 = view_126 = None
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(permute_350, [1, 0]);  permute_350 = None
    view_450: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_80, [4, 512, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_451: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_450, [4, 512, 12, 64]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_352: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_109: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
    view_452: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_109, [48, 512, 64]);  clone_109 = None
    bmm_48: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_353, view_452);  permute_353 = None
    bmm_49: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_452, permute_354);  view_452 = permute_354 = None
    view_453: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_48, [4, 12, 512, 64]);  bmm_48 = None
    view_454: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_49, [4, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_265: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_454, alias_18);  view_454 = None
    sum_120: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_265, [-1], True)
    mul_266: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_18, sum_120);  alias_18 = sum_120 = None
    sub_90: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_265, mul_266);  mul_265 = mul_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_45: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_90, 8.0);  sub_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_455: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_45, [48, 512, 512]);  div_45 = None
    bmm_50: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_355, view_455);  permute_355 = None
    bmm_51: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_455, permute_356);  view_455 = permute_356 = None
    view_456: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_50, [4, 12, 64, 512]);  bmm_50 = None
    view_457: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_51, [4, 12, 512, 64]);  bmm_51 = None
    permute_357: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 1, 3, 2]);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_358: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_110: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_458: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_110, [4, 512, 768]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_111: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_459: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_111, [4, 512, 768]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_460: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_459, [2048, 768]);  view_459 = None
    mm_82: "f32[2048, 768]" = torch.ops.aten.mm.default(view_460, permute_360);  permute_360 = None
    permute_361: "f32[768, 2048]" = torch.ops.aten.permute.default(view_460, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_460, [0], True);  view_460 = None
    view_461: "f32[768]" = torch.ops.aten.reshape.default(sum_121, [768]);  sum_121 = None
    permute_363: "f32[768, 768]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
    view_462: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_82, [4, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_144: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_263, view_462);  mul_263 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_364: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_357, [0, 2, 1, 3]);  permute_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_463: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_364, [4, 512, 768]);  permute_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_112: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_463, memory_format = torch.contiguous_format);  view_463 = None
    view_464: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_112, [2048, 768]);  clone_112 = None
    mm_84: "f32[2048, 768]" = torch.ops.aten.mm.default(view_464, permute_365);  permute_365 = None
    permute_366: "f32[768, 2048]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_366, view_110);  permute_366 = None
    permute_367: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[768]" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
    permute_368: "f32[768, 768]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
    view_466: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_84, [4, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_145: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_144, view_466);  add_144 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_467: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_458, [2048, 768]);  view_458 = None
    mm_86: "f32[2048, 768]" = torch.ops.aten.mm.default(view_467, permute_369);  permute_369 = None
    permute_370: "f32[768, 2048]" = torch.ops.aten.permute.default(view_467, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_370, view_110);  permute_370 = view_110 = None
    permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0], True);  view_467 = None
    view_468: "f32[768]" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
    permute_372: "f32[768, 768]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
    view_469: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_86, [4, 512, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_146: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_469);  add_145 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_268: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_84);  primals_84 = None
    mul_269: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_268, 768)
    sum_124: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [2], True)
    mul_270: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_268, mul_36);  mul_268 = None
    sum_125: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True);  mul_270 = None
    mul_271: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_36, sum_125);  sum_125 = None
    sub_92: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_269, sum_124);  mul_269 = sum_124 = None
    sub_93: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_271);  sub_92 = mul_271 = None
    mul_272: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_93);  div_46 = sub_93 = None
    mul_273: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, mul_36);  mul_36 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 1]);  mul_273 = None
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1]);  add_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_470: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_272, [2048, 768])
    mm_88: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_470, permute_373);  permute_373 = None
    permute_374: "f32[768, 2048]" = torch.ops.aten.permute.default(view_470, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_374, view_108);  permute_374 = view_108 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
    view_471: "f32[768]" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
    permute_376: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
    view_472: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_88, [4, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_275: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_276: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_277: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_276, -0.5);  mul_276 = None
    exp_20: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_277);  mul_277 = None
    mul_278: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_279: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_278);  view_107 = mul_278 = None
    add_148: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None
    mul_280: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_472, add_148);  view_472 = add_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_473: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_280, [2048, 3072]);  mul_280 = None
    mm_90: "f32[2048, 768]" = torch.ops.aten.mm.default(view_473, permute_377);  permute_377 = None
    permute_378: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_378, view_106);  permute_378 = view_106 = None
    permute_379: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_129: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_473, [0], True);  view_473 = None
    view_474: "f32[3072]" = torch.ops.aten.reshape.default(sum_129, [3072]);  sum_129 = None
    permute_380: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
    view_475: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_90, [4, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_149: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_272, view_475);  mul_272 = view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_282: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, primals_78);  primals_78 = None
    mul_283: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, 768)
    sum_130: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True)
    mul_284: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_282, mul_31);  mul_282 = None
    sum_131: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_284, [2], True);  mul_284 = None
    mul_285: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_31, sum_131);  sum_131 = None
    sub_95: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_283, sum_130);  mul_283 = sum_130 = None
    sub_96: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_285);  sub_95 = mul_285 = None
    mul_286: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_96);  div_47 = sub_96 = None
    mul_287: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_149, mul_31);  mul_31 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1]);  mul_287 = None
    sum_133: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_149, [0, 1]);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_476: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_286, [2048, 768])
    mm_92: "f32[2048, 768]" = torch.ops.aten.mm.default(view_476, permute_381);  permute_381 = None
    permute_382: "f32[768, 2048]" = torch.ops.aten.permute.default(view_476, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_382, view_104);  permute_382 = view_104 = None
    permute_383: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_134: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_476, [0], True);  view_476 = None
    view_477: "f32[768]" = torch.ops.aten.reshape.default(sum_134, [768]);  sum_134 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
    view_478: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_92, [4, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_479: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_478, [4, 512, 12, 64]);  view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_385: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_113: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_480: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_113, [48, 512, 64]);  clone_113 = None
    bmm_52: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_386, view_480);  permute_386 = None
    bmm_53: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_480, permute_387);  view_480 = permute_387 = None
    view_481: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_52, [4, 12, 512, 64]);  bmm_52 = None
    view_482: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_53, [4, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_288: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_482, alias_19);  view_482 = None
    sum_135: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [-1], True)
    mul_289: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_19, sum_135);  alias_19 = sum_135 = None
    sub_97: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_288, mul_289);  mul_288 = mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_48: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_97, 8.0);  sub_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_483: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_48, [48, 512, 512]);  div_48 = None
    bmm_54: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_388, view_483);  permute_388 = None
    bmm_55: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_483, permute_389);  view_483 = permute_389 = None
    view_484: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_54, [4, 12, 64, 512]);  bmm_54 = None
    view_485: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_55, [4, 12, 512, 64]);  bmm_55 = None
    permute_390: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 1, 3, 2]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_391: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_114: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_486: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_114, [4, 512, 768]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_481, [0, 2, 1, 3]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_115: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_487: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_115, [4, 512, 768]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_488: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_487, [2048, 768]);  view_487 = None
    mm_94: "f32[2048, 768]" = torch.ops.aten.mm.default(view_488, permute_393);  permute_393 = None
    permute_394: "f32[768, 2048]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[768]" = torch.ops.aten.reshape.default(sum_136, [768]);  sum_136 = None
    permute_396: "f32[768, 768]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
    view_490: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_94, [4, 512, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_150: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_286, view_490);  mul_286 = view_490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_397: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_390, [0, 2, 1, 3]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_491: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_397, [4, 512, 768]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_116: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_491, memory_format = torch.contiguous_format);  view_491 = None
    view_492: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_116, [2048, 768]);  clone_116 = None
    mm_96: "f32[2048, 768]" = torch.ops.aten.mm.default(view_492, permute_398);  permute_398 = None
    permute_399: "f32[768, 2048]" = torch.ops.aten.permute.default(view_492, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_399, view_88);  permute_399 = None
    permute_400: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_492, [0], True);  view_492 = None
    view_493: "f32[768]" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
    permute_401: "f32[768, 768]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_494: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_96, [4, 512, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_151: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_150, view_494);  add_150 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_495: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_486, [2048, 768]);  view_486 = None
    mm_98: "f32[2048, 768]" = torch.ops.aten.mm.default(view_495, permute_402);  permute_402 = None
    permute_403: "f32[768, 2048]" = torch.ops.aten.permute.default(view_495, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_403, view_88);  permute_403 = view_88 = None
    permute_404: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_495, [0], True);  view_495 = None
    view_496: "f32[768]" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
    permute_405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_404, [1, 0]);  permute_404 = None
    view_497: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_98, [4, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_152: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_151, view_497);  add_151 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_291: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_68);  primals_68 = None
    mul_292: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, 768)
    sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True)
    mul_293: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_29);  mul_291 = None
    sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    mul_294: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_29, sum_140);  sum_140 = None
    sub_99: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_292, sum_139);  mul_292 = sum_139 = None
    sub_100: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_99, mul_294);  sub_99 = mul_294 = None
    mul_295: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_49, sub_100);  div_49 = sub_100 = None
    mul_296: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, mul_29);  mul_29 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 1]);  mul_296 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_498: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_295, [2048, 768])
    mm_100: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_498, permute_406);  permute_406 = None
    permute_407: "f32[768, 2048]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_101: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_407, view_86);  permute_407 = view_86 = None
    permute_408: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_498, [0], True);  view_498 = None
    view_499: "f32[768]" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
    permute_409: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_408, [1, 0]);  permute_408 = None
    view_500: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_100, [4, 512, 3072]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_298: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_299: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_300: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_299, -0.5);  mul_299 = None
    exp_21: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_300);  mul_300 = None
    mul_301: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_302: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_301);  view_85 = mul_301 = None
    add_154: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_298, mul_302);  mul_298 = mul_302 = None
    mul_303: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_500, add_154);  view_500 = add_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_501: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_303, [2048, 3072]);  mul_303 = None
    mm_102: "f32[2048, 768]" = torch.ops.aten.mm.default(view_501, permute_410);  permute_410 = None
    permute_411: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_501, [1, 0])
    mm_103: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_411, view_84);  permute_411 = view_84 = None
    permute_412: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_144: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_501, [0], True);  view_501 = None
    view_502: "f32[3072]" = torch.ops.aten.reshape.default(sum_144, [3072]);  sum_144 = None
    permute_413: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
    view_503: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_102, [4, 512, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_155: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_295, view_503);  mul_295 = view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_305: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_62);  primals_62 = None
    mul_306: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, 768)
    sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True)
    mul_307: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_305, mul_24);  mul_305 = None
    sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True);  mul_307 = None
    mul_308: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_24, sum_146);  sum_146 = None
    sub_102: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_306, sum_145);  mul_306 = sum_145 = None
    sub_103: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_308);  sub_102 = mul_308 = None
    mul_309: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_103);  div_50 = sub_103 = None
    mul_310: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_24);  mul_24 = None
    sum_147: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 1]);  mul_310 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_504: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_309, [2048, 768])
    mm_104: "f32[2048, 768]" = torch.ops.aten.mm.default(view_504, permute_414);  permute_414 = None
    permute_415: "f32[768, 2048]" = torch.ops.aten.permute.default(view_504, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_415, view_82);  permute_415 = view_82 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_149: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True);  view_504 = None
    view_505: "f32[768]" = torch.ops.aten.reshape.default(sum_149, [768]);  sum_149 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
    view_506: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_104, [4, 512, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_507: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_506, [4, 512, 12, 64]);  view_506 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_418: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_117: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
    view_508: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_117, [48, 512, 64]);  clone_117 = None
    bmm_56: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_419, view_508);  permute_419 = None
    bmm_57: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_508, permute_420);  view_508 = permute_420 = None
    view_509: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_56, [4, 12, 512, 64]);  bmm_56 = None
    view_510: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_57, [4, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_311: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_510, alias_20);  view_510 = None
    sum_150: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [-1], True)
    mul_312: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_20, sum_150);  alias_20 = sum_150 = None
    sub_104: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_51: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_104, 8.0);  sub_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_511: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_51, [48, 512, 512]);  div_51 = None
    bmm_58: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_421, view_511);  permute_421 = None
    bmm_59: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_511, permute_422);  view_511 = permute_422 = None
    view_512: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_58, [4, 12, 64, 512]);  bmm_58 = None
    view_513: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_59, [4, 12, 512, 64]);  bmm_59 = None
    permute_423: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 1, 3, 2]);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_513, [0, 2, 1, 3]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_118: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_514: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_118, [4, 512, 768]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_509, [0, 2, 1, 3]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_119: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_515: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_119, [4, 512, 768]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_516: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_515, [2048, 768]);  view_515 = None
    mm_106: "f32[2048, 768]" = torch.ops.aten.mm.default(view_516, permute_426);  permute_426 = None
    permute_427: "f32[768, 2048]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_151: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
    view_517: "f32[768]" = torch.ops.aten.reshape.default(sum_151, [768]);  sum_151 = None
    permute_429: "f32[768, 768]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_518: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_106, [4, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_156: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_309, view_518);  mul_309 = view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_430: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_423, [0, 2, 1, 3]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_519: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_430, [4, 512, 768]);  permute_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_120: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_519, memory_format = torch.contiguous_format);  view_519 = None
    view_520: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_120, [2048, 768]);  clone_120 = None
    mm_108: "f32[2048, 768]" = torch.ops.aten.mm.default(view_520, permute_431);  permute_431 = None
    permute_432: "f32[768, 2048]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_432, view_66);  permute_432 = None
    permute_433: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
    view_521: "f32[768]" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
    permute_434: "f32[768, 768]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
    view_522: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_108, [4, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_157: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_156, view_522);  add_156 = view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_523: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_514, [2048, 768]);  view_514 = None
    mm_110: "f32[2048, 768]" = torch.ops.aten.mm.default(view_523, permute_435);  permute_435 = None
    permute_436: "f32[768, 2048]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_436, view_66);  permute_436 = view_66 = None
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
    permute_438: "f32[768, 768]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
    view_525: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_110, [4, 512, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_158: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_157, view_525);  add_157 = view_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_314: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, primals_52);  primals_52 = None
    mul_315: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_314, 768)
    sum_154: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True)
    mul_316: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_314, mul_22);  mul_314 = None
    sum_155: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True);  mul_316 = None
    mul_317: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_155);  sum_155 = None
    sub_106: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_315, sum_154);  mul_315 = sum_154 = None
    sub_107: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_106, mul_317);  sub_106 = mul_317 = None
    mul_318: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_107);  div_52 = sub_107 = None
    mul_319: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_158, mul_22);  mul_22 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 1]);  mul_319 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_158, [0, 1]);  add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_526: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_318, [2048, 768])
    mm_112: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_526, permute_439);  permute_439 = None
    permute_440: "f32[768, 2048]" = torch.ops.aten.permute.default(view_526, [1, 0])
    mm_113: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_440, view_64);  permute_440 = view_64 = None
    permute_441: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
    view_527: "f32[768]" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
    permute_442: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
    view_528: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_112, [4, 512, 3072]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_321: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_322: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_323: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_322, -0.5);  mul_322 = None
    exp_22: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_323);  mul_323 = None
    mul_324: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_325: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_324);  view_63 = mul_324 = None
    add_160: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_321, mul_325);  mul_321 = mul_325 = None
    mul_326: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_528, add_160);  view_528 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_529: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_326, [2048, 3072]);  mul_326 = None
    mm_114: "f32[2048, 768]" = torch.ops.aten.mm.default(view_529, permute_443);  permute_443 = None
    permute_444: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_529, [1, 0])
    mm_115: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_444, view_62);  permute_444 = view_62 = None
    permute_445: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_159: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
    view_530: "f32[3072]" = torch.ops.aten.reshape.default(sum_159, [3072]);  sum_159 = None
    permute_446: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
    view_531: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_114, [4, 512, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_161: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_318, view_531);  mul_318 = view_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_328: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, primals_46);  primals_46 = None
    mul_329: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, 768)
    sum_160: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True)
    mul_330: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_328, mul_17);  mul_328 = None
    sum_161: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
    mul_331: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_161);  sum_161 = None
    sub_109: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_329, sum_160);  mul_329 = sum_160 = None
    sub_110: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_109, mul_331);  sub_109 = mul_331 = None
    mul_332: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_53, sub_110);  div_53 = sub_110 = None
    mul_333: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_161, mul_17);  mul_17 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1]);  mul_333 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_161, [0, 1]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_532: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_332, [2048, 768])
    mm_116: "f32[2048, 768]" = torch.ops.aten.mm.default(view_532, permute_447);  permute_447 = None
    permute_448: "f32[768, 2048]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_60);  permute_448 = view_60 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[768]" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    view_534: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_116, [4, 512, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_535: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_534, [4, 512, 12, 64]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_451: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_121: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None
    view_536: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_121, [48, 512, 64]);  clone_121 = None
    bmm_60: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_452, view_536);  permute_452 = None
    bmm_61: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_536, permute_453);  view_536 = permute_453 = None
    view_537: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_60, [4, 12, 512, 64]);  bmm_60 = None
    view_538: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_61, [4, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_334: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_538, alias_21);  view_538 = None
    sum_165: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [-1], True)
    mul_335: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_21, sum_165);  alias_21 = sum_165 = None
    sub_111: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_54: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_111, 8.0);  sub_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_539: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_54, [48, 512, 512]);  div_54 = None
    bmm_62: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_454, view_539);  permute_454 = None
    bmm_63: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_539, permute_455);  view_539 = permute_455 = None
    view_540: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_62, [4, 12, 64, 512]);  bmm_62 = None
    view_541: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_63, [4, 12, 512, 64]);  bmm_63 = None
    permute_456: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 1, 3, 2]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_457: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_541, [0, 2, 1, 3]);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_122: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_542: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_122, [4, 512, 768]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_123: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_543: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_123, [4, 512, 768]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_544: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_543, [2048, 768]);  view_543 = None
    mm_118: "f32[2048, 768]" = torch.ops.aten.mm.default(view_544, permute_459);  permute_459 = None
    permute_460: "f32[768, 2048]" = torch.ops.aten.permute.default(view_544, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
    view_545: "f32[768]" = torch.ops.aten.reshape.default(sum_166, [768]);  sum_166 = None
    permute_462: "f32[768, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_546: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_118, [4, 512, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_162: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_332, view_546);  mul_332 = view_546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_463: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_456, [0, 2, 1, 3]);  permute_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_547: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_463, [4, 512, 768]);  permute_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_124: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_547, memory_format = torch.contiguous_format);  view_547 = None
    view_548: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_124, [2048, 768]);  clone_124 = None
    mm_120: "f32[2048, 768]" = torch.ops.aten.mm.default(view_548, permute_464);  permute_464 = None
    permute_465: "f32[768, 2048]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_465, view_44);  permute_465 = None
    permute_466: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
    view_549: "f32[768]" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
    permute_467: "f32[768, 768]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_550: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_120, [4, 512, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_163: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_162, view_550);  add_162 = view_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_551: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_542, [2048, 768]);  view_542 = None
    mm_122: "f32[2048, 768]" = torch.ops.aten.mm.default(view_551, permute_468);  permute_468 = None
    permute_469: "f32[768, 2048]" = torch.ops.aten.permute.default(view_551, [1, 0])
    mm_123: "f32[768, 768]" = torch.ops.aten.mm.default(permute_469, view_44);  permute_469 = view_44 = None
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_168: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
    view_552: "f32[768]" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
    permute_471: "f32[768, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_553: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_122, [4, 512, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_164: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_163, view_553);  add_163 = view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_337: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_36);  primals_36 = None
    mul_338: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_337, 768)
    sum_169: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
    mul_339: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_337, mul_15);  mul_337 = None
    sum_170: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
    mul_340: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_170);  sum_170 = None
    sub_113: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_338, sum_169);  mul_338 = sum_169 = None
    sub_114: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_113, mul_340);  sub_113 = mul_340 = None
    mul_341: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_55, sub_114);  div_55 = sub_114 = None
    mul_342: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_15);  mul_15 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_554: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_341, [2048, 768])
    mm_124: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_554, permute_472);  permute_472 = None
    permute_473: "f32[768, 2048]" = torch.ops.aten.permute.default(view_554, [1, 0])
    mm_125: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_473, view_42);  permute_473 = view_42 = None
    permute_474: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_173: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
    view_555: "f32[768]" = torch.ops.aten.reshape.default(sum_173, [768]);  sum_173 = None
    permute_475: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_556: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_124, [4, 512, 3072]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_344: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_345: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_346: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_345, -0.5);  mul_345 = None
    exp_23: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_346);  mul_346 = None
    mul_347: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_348: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_347);  view_41 = mul_347 = None
    add_166: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_344, mul_348);  mul_344 = mul_348 = None
    mul_349: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_556, add_166);  view_556 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_557: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_349, [2048, 3072]);  mul_349 = None
    mm_126: "f32[2048, 768]" = torch.ops.aten.mm.default(view_557, permute_476);  permute_476 = None
    permute_477: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_127: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_477, view_40);  permute_477 = view_40 = None
    permute_478: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_174: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
    view_558: "f32[3072]" = torch.ops.aten.reshape.default(sum_174, [3072]);  sum_174 = None
    permute_479: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
    view_559: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_126, [4, 512, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_167: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_341, view_559);  mul_341 = view_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_351: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_30);  primals_30 = None
    mul_352: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_351, 768)
    sum_175: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True)
    mul_353: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_351, mul_10);  mul_351 = None
    sum_176: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True);  mul_353 = None
    mul_354: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_176);  sum_176 = None
    sub_116: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_352, sum_175);  mul_352 = sum_175 = None
    sub_117: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_116, mul_354);  sub_116 = mul_354 = None
    mul_355: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_117);  div_56 = sub_117 = None
    mul_356: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_10);  mul_10 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1]);  mul_356 = None
    sum_178: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_560: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_355, [2048, 768])
    mm_128: "f32[2048, 768]" = torch.ops.aten.mm.default(view_560, permute_480);  permute_480 = None
    permute_481: "f32[768, 2048]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_481, view_38);  permute_481 = view_38 = None
    permute_482: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[768]" = torch.ops.aten.reshape.default(sum_179, [768]);  sum_179 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_482, [1, 0]);  permute_482 = None
    view_562: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_128, [4, 512, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_563: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_562, [4, 512, 12, 64]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_484: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_125: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None
    view_564: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_125, [48, 512, 64]);  clone_125 = None
    bmm_64: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_485, view_564);  permute_485 = None
    bmm_65: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_564, permute_486);  view_564 = permute_486 = None
    view_565: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_64, [4, 12, 512, 64]);  bmm_64 = None
    view_566: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_65, [4, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_357: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_566, alias_22);  view_566 = None
    sum_180: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [-1], True)
    mul_358: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_22, sum_180);  alias_22 = sum_180 = None
    sub_118: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_57: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_118, 8.0);  sub_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_567: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_57, [48, 512, 512]);  div_57 = None
    bmm_66: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_487, view_567);  permute_487 = None
    bmm_67: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_567, permute_488);  view_567 = permute_488 = None
    view_568: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_66, [4, 12, 64, 512]);  bmm_66 = None
    view_569: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_67, [4, 12, 512, 64]);  bmm_67 = None
    permute_489: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_568, [0, 1, 3, 2]);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_490: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_126: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_570: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_126, [4, 512, 768]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_127: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_571: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_127, [4, 512, 768]);  clone_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_572: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_571, [2048, 768]);  view_571 = None
    mm_130: "f32[2048, 768]" = torch.ops.aten.mm.default(view_572, permute_492);  permute_492 = None
    permute_493: "f32[768, 2048]" = torch.ops.aten.permute.default(view_572, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
    view_573: "f32[768]" = torch.ops.aten.reshape.default(sum_181, [768]);  sum_181 = None
    permute_495: "f32[768, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_574: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_130, [4, 512, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_168: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_355, view_574);  mul_355 = view_574 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_496: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_489, [0, 2, 1, 3]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_575: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_496, [4, 512, 768]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_128: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_575, memory_format = torch.contiguous_format);  view_575 = None
    view_576: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_128, [2048, 768]);  clone_128 = None
    mm_132: "f32[2048, 768]" = torch.ops.aten.mm.default(view_576, permute_497);  permute_497 = None
    permute_498: "f32[768, 2048]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_498, view_22);  permute_498 = None
    permute_499: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[768]" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
    permute_500: "f32[768, 768]" = torch.ops.aten.permute.default(permute_499, [1, 0]);  permute_499 = None
    view_578: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_132, [4, 512, 768]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_169: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_168, view_578);  add_168 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_579: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_570, [2048, 768]);  view_570 = None
    mm_134: "f32[2048, 768]" = torch.ops.aten.mm.default(view_579, permute_501);  permute_501 = None
    permute_502: "f32[768, 2048]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_135: "f32[768, 768]" = torch.ops.aten.mm.default(permute_502, view_22);  permute_502 = view_22 = None
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_183: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[768]" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
    permute_504: "f32[768, 768]" = torch.ops.aten.permute.default(permute_503, [1, 0]);  permute_503 = None
    view_581: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_134, [4, 512, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_170: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_581);  add_169 = view_581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:466, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_360: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, primals_20);  primals_20 = None
    mul_361: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_360, 768)
    sum_184: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True)
    mul_362: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_360, mul_8);  mul_360 = None
    sum_185: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [2], True);  mul_362 = None
    mul_363: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_8, sum_185);  sum_185 = None
    sub_120: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_361, sum_184);  mul_361 = sum_184 = None
    sub_121: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_120, mul_363);  sub_120 = mul_363 = None
    mul_364: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_58, sub_121);  div_58 = sub_121 = None
    mul_365: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_170, mul_8);  mul_8 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 1]);  mul_365 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_170, [0, 1]);  add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:464, code: hidden_states = self.dense(hidden_states)
    view_582: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_364, [2048, 768])
    mm_136: "f32[2048, 3072]" = torch.ops.aten.mm.default(view_582, permute_505);  permute_505 = None
    permute_506: "f32[768, 2048]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_137: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_506, view_20);  permute_506 = view_20 = None
    permute_507: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
    permute_508: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_507, [1, 0]);  permute_507 = None
    view_584: "f32[4, 512, 3072]" = torch.ops.aten.reshape.default(mm_136, [4, 512, 3072]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_367: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_368: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_369: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_368, -0.5);  mul_368 = None
    exp_24: "f32[4, 512, 3072]" = torch.ops.aten.exp.default(mul_369);  mul_369 = None
    mul_370: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_371: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_370);  view_19 = mul_370 = None
    add_172: "f32[4, 512, 3072]" = torch.ops.aten.add.Tensor(mul_367, mul_371);  mul_367 = mul_371 = None
    mul_372: "f32[4, 512, 3072]" = torch.ops.aten.mul.Tensor(view_584, add_172);  view_584 = add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    view_585: "f32[2048, 3072]" = torch.ops.aten.reshape.default(mul_372, [2048, 3072]);  mul_372 = None
    mm_138: "f32[2048, 768]" = torch.ops.aten.mm.default(view_585, permute_509);  permute_509 = None
    permute_510: "f32[3072, 2048]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_139: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_510, view_18);  permute_510 = view_18 = None
    permute_511: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_189: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[3072]" = torch.ops.aten.reshape.default(sum_189, [3072]);  sum_189 = None
    permute_512: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_511, [1, 0]);  permute_511 = None
    view_587: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_138, [4, 512, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:451, code: hidden_states = self.dense(hidden_states)
    add_173: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_364, view_587);  mul_364 = view_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:388, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_374: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, primals_14);  primals_14 = None
    mul_375: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, 768)
    sum_190: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True)
    mul_376: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_374, mul_3);  mul_374 = None
    sum_191: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
    mul_377: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_3, sum_191);  sum_191 = None
    sub_123: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_375, sum_190);  mul_375 = sum_190 = None
    sub_124: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_123, mul_377);  sub_123 = mul_377 = None
    mul_378: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_59, sub_124);  div_59 = sub_124 = None
    mul_379: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_173, mul_3);  mul_3 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1]);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:386, code: hidden_states = self.dense(hidden_states)
    view_588: "f32[2048, 768]" = torch.ops.aten.reshape.default(mul_378, [2048, 768])
    mm_140: "f32[2048, 768]" = torch.ops.aten.mm.default(view_588, permute_513);  permute_513 = None
    permute_514: "f32[768, 2048]" = torch.ops.aten.permute.default(view_588, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_514, view_16);  permute_514 = view_16 = None
    permute_515: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_194: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
    view_589: "f32[768]" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(permute_515, [1, 0]);  permute_515 = None
    view_590: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_140, [4, 512, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:369, code: context_layer = context_layer.view(new_context_layer_shape)
    view_591: "f32[4, 512, 12, 64]" = torch.ops.aten.reshape.default(view_590, [4, 512, 12, 64]);  view_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:367, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_517: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_591, [0, 2, 1, 3]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:365, code: context_layer = torch.matmul(attention_probs, value_layer)
    clone_129: "f32[4, 12, 512, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None
    view_592: "f32[48, 512, 64]" = torch.ops.aten.reshape.default(clone_129, [48, 512, 64]);  clone_129 = None
    bmm_68: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(permute_518, view_592);  permute_518 = None
    bmm_69: "f32[48, 512, 512]" = torch.ops.aten.bmm.default(view_592, permute_519);  view_592 = permute_519 = None
    view_593: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_68, [4, 12, 512, 64]);  bmm_68 = None
    view_594: "f32[4, 12, 512, 512]" = torch.ops.aten.reshape.default(bmm_69, [4, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:355, code: attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    mul_380: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(view_594, alias_23);  view_594 = None
    sum_195: "f32[4, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [-1], True)
    mul_381: "f32[4, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_23, sum_195);  alias_23 = sum_195 = None
    sub_125: "f32[4, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_380, mul_381);  mul_380 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:349, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    div_60: "f32[4, 12, 512, 512]" = torch.ops.aten.div.Tensor(sub_125, 8.0);  sub_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:325, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_595: "f32[48, 512, 512]" = torch.ops.aten.reshape.default(div_60, [48, 512, 512]);  div_60 = None
    bmm_70: "f32[48, 64, 512]" = torch.ops.aten.bmm.default(permute_520, view_595);  permute_520 = None
    bmm_71: "f32[48, 512, 64]" = torch.ops.aten.bmm.default(view_595, permute_521);  view_595 = permute_521 = None
    view_596: "f32[4, 12, 64, 512]" = torch.ops.aten.reshape.default(bmm_70, [4, 12, 64, 512]);  bmm_70 = None
    view_597: "f32[4, 12, 512, 64]" = torch.ops.aten.reshape.default(bmm_71, [4, 12, 512, 64]);  bmm_71 = None
    permute_522: "f32[4, 12, 512, 64]" = torch.ops.aten.permute.default(view_596, [0, 1, 3, 2]);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_523: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_130: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_598: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_130, [4, 512, 768]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(view_593, [0, 2, 1, 3]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    clone_131: "f32[4, 512, 12, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_599: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(clone_131, [4, 512, 768]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_600: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_599, [2048, 768]);  view_599 = None
    mm_142: "f32[2048, 768]" = torch.ops.aten.mm.default(view_600, permute_525);  permute_525 = None
    permute_526: "f32[768, 2048]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_196: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[768]" = torch.ops.aten.reshape.default(sum_196, [768]);  sum_196 = None
    permute_528: "f32[768, 768]" = torch.ops.aten.permute.default(permute_527, [1, 0]);  permute_527 = None
    view_602: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_142, [4, 512, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:309, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_174: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(mul_378, view_602);  mul_378 = view_602 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:274, code: return x.permute(0, 2, 1, 3)
    permute_529: "f32[4, 512, 12, 64]" = torch.ops.aten.permute.default(permute_522, [0, 2, 1, 3]);  permute_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:273, code: x = x.view(new_x_shape)
    view_603: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(permute_529, [4, 512, 768]);  permute_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    clone_132: "f32[4, 512, 768]" = torch.ops.aten.clone.default(view_603, memory_format = torch.contiguous_format);  view_603 = None
    view_604: "f32[2048, 768]" = torch.ops.aten.reshape.default(clone_132, [2048, 768]);  clone_132 = None
    mm_144: "f32[2048, 768]" = torch.ops.aten.mm.default(view_604, permute_530);  permute_530 = None
    permute_531: "f32[768, 2048]" = torch.ops.aten.permute.default(view_604, [1, 0])
    mm_145: "f32[768, 768]" = torch.ops.aten.mm.default(permute_531, view);  permute_531 = None
    permute_532: "f32[768, 768]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_197: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True);  view_604 = None
    view_605: "f32[768]" = torch.ops.aten.reshape.default(sum_197, [768]);  sum_197 = None
    permute_533: "f32[768, 768]" = torch.ops.aten.permute.default(permute_532, [1, 0]);  permute_532 = None
    view_606: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_144, [4, 512, 768]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:308, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_175: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_174, view_606);  add_174 = view_606 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    view_607: "f32[2048, 768]" = torch.ops.aten.reshape.default(view_598, [2048, 768]);  view_598 = None
    mm_146: "f32[2048, 768]" = torch.ops.aten.mm.default(view_607, permute_534);  permute_534 = None
    permute_535: "f32[768, 2048]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_147: "f32[768, 768]" = torch.ops.aten.mm.default(permute_535, view);  permute_535 = view = None
    permute_536: "f32[768, 768]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[768]" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
    permute_537: "f32[768, 768]" = torch.ops.aten.permute.default(permute_536, [1, 0]);  permute_536 = None
    view_609: "f32[4, 512, 768]" = torch.ops.aten.reshape.default(mm_146, [4, 512, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:286, code: mixed_query_layer = self.query(hidden_states)
    add_176: "f32[4, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_609);  add_175 = view_609 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:239, code: embeddings = self.LayerNorm(embeddings)
    mul_383: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_4);  primals_4 = None
    mul_384: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_383, 768)
    sum_199: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_383, mul_1);  mul_383 = None
    sum_200: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_200);  sum_200 = None
    sub_127: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(mul_384, sum_199);  mul_384 = sum_199 = None
    sub_128: "f32[4, 512, 768]" = torch.ops.aten.sub.Tensor(sub_127, mul_386);  sub_127 = mul_386 = None
    mul_387: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_128);  div_61 = sub_128 = None
    mul_388: "f32[4, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, mul_1);  mul_1 = None
    sum_201: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_202: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1]);  add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:238, code: embeddings += position_embeddings
    sum_203: "f32[1, 512, 768]" = torch.ops.aten.sum.dim_IntList(mul_387, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:237, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_2: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_2, full_default_1, sum_203);  unsqueeze_2 = sum_203 = None
    full_default_2: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_2, [slice_4], where, True);  full_default_2 = slice_4 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:233, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[4, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_3: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_1: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, mul_387);  unsqueeze_3 = None
    full_default_4: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_4, [expand], where_1, True);  full_default_4 = expand = where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/bert/modeling_bert.py:232, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_206, 0)
    unsqueeze_4: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_2: "f32[4, 512, 768]" = torch.ops.aten.where.self(unsqueeze_4, full_default_1, mul_387);  unsqueeze_4 = full_default_1 = mul_387 = None
    full_default_6: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 768]" = torch.ops.prims._unsafe_index_put_.default(full_default_6, [primals_206], where_2, True);  full_default_6 = primals_206 = where_2 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_201, sum_202, permute_537, view_608, permute_533, view_605, permute_528, view_601, permute_516, view_589, sum_192, sum_193, permute_512, view_586, permute_508, view_583, sum_186, sum_187, permute_504, view_580, permute_500, view_577, permute_495, view_573, permute_483, view_561, sum_177, sum_178, permute_479, view_558, permute_475, view_555, sum_171, sum_172, permute_471, view_552, permute_467, view_549, permute_462, view_545, permute_450, view_533, sum_162, sum_163, permute_446, view_530, permute_442, view_527, sum_156, sum_157, permute_438, view_524, permute_434, view_521, permute_429, view_517, permute_417, view_505, sum_147, sum_148, permute_413, view_502, permute_409, view_499, sum_141, sum_142, permute_405, view_496, permute_401, view_493, permute_396, view_489, permute_384, view_477, sum_132, sum_133, permute_380, view_474, permute_376, view_471, sum_126, sum_127, permute_372, view_468, permute_368, view_465, permute_363, view_461, permute_351, view_449, sum_117, sum_118, permute_347, view_446, permute_343, view_443, sum_111, sum_112, permute_339, view_440, permute_335, view_437, permute_330, view_433, permute_318, view_421, sum_102, sum_103, permute_314, view_418, permute_310, view_415, sum_96, sum_97, permute_306, view_412, permute_302, view_409, permute_297, view_405, permute_285, view_393, sum_87, sum_88, permute_281, view_390, permute_277, view_387, sum_81, sum_82, permute_273, view_384, permute_269, view_381, permute_264, view_377, permute_252, view_365, sum_72, sum_73, permute_248, view_362, permute_244, view_359, sum_66, sum_67, permute_240, view_356, permute_236, view_353, permute_231, view_349, permute_219, view_337, sum_57, sum_58, permute_215, view_334, permute_211, view_331, sum_51, sum_52, permute_207, view_328, permute_203, view_325, permute_198, view_321, permute_186, view_309, sum_42, sum_43, permute_182, view_306, permute_178, view_303, sum_36, sum_37, permute_174, view_300, permute_170, view_297, permute_165, view_293, permute_153, view_281, sum_27, sum_28, permute_149, view_278, permute_145, view_275, sum_21, sum_22, permute_141, view_272, sum_16, sum_17, permute_137, view_269, None, None, None]
    