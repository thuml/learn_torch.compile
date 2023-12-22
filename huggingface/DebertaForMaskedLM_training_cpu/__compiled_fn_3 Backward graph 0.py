from __future__ import annotations



def forward(self, primals_1: "f32[768]", primals_5: "f32[768]", primals_7: "f32[768]", primals_11: "f32[768]", primals_13: "f32[768]", primals_17: "f32[768]", primals_19: "f32[768]", primals_23: "f32[768]", primals_25: "f32[768]", primals_29: "f32[768]", primals_31: "f32[768]", primals_35: "f32[768]", primals_37: "f32[768]", primals_41: "f32[768]", primals_43: "f32[768]", primals_47: "f32[768]", primals_49: "f32[768]", primals_53: "f32[768]", primals_55: "f32[768]", primals_59: "f32[768]", primals_61: "f32[768]", primals_65: "f32[768]", primals_67: "f32[768]", primals_71: "f32[768]", primals_73: "f32[768]", primals_163: "f32[768]", primals_168: "i64[1, 512]", primals_169: "i64[1, 512]", slice_1: "i64[1, 512]", sub: "f32[1, 512, 768]", sqrt: "f32[1, 512, 1]", convert_element_type: "b8[1, 512, 768]", view: "f32[512, 768]", convert_element_type_2: "b8[1, 12, 512, 512]", view_12: "f32[512, 768]", convert_element_type_3: "b8[1, 512, 768]", sub_6: "f32[1, 512, 768]", sqrt_2: "f32[1, 512, 1]", view_14: "f32[512, 768]", addmm_1: "f32[512, 3072]", view_16: "f32[512, 3072]", convert_element_type_4: "b8[1, 512, 768]", sub_9: "f32[1, 512, 768]", sqrt_3: "f32[1, 512, 1]", view_18: "f32[512, 768]", convert_element_type_6: "b8[1, 12, 512, 512]", view_30: "f32[512, 768]", convert_element_type_7: "b8[1, 512, 768]", sub_14: "f32[1, 512, 768]", sqrt_5: "f32[1, 512, 1]", view_32: "f32[512, 768]", addmm_4: "f32[512, 3072]", view_34: "f32[512, 3072]", convert_element_type_8: "b8[1, 512, 768]", sub_17: "f32[1, 512, 768]", sqrt_6: "f32[1, 512, 1]", view_36: "f32[512, 768]", convert_element_type_10: "b8[1, 12, 512, 512]", view_48: "f32[512, 768]", convert_element_type_11: "b8[1, 512, 768]", sub_22: "f32[1, 512, 768]", sqrt_8: "f32[1, 512, 1]", view_50: "f32[512, 768]", addmm_7: "f32[512, 3072]", view_52: "f32[512, 3072]", convert_element_type_12: "b8[1, 512, 768]", sub_25: "f32[1, 512, 768]", sqrt_9: "f32[1, 512, 1]", view_54: "f32[512, 768]", convert_element_type_14: "b8[1, 12, 512, 512]", view_66: "f32[512, 768]", convert_element_type_15: "b8[1, 512, 768]", sub_30: "f32[1, 512, 768]", sqrt_11: "f32[1, 512, 1]", view_68: "f32[512, 768]", addmm_10: "f32[512, 3072]", view_70: "f32[512, 3072]", convert_element_type_16: "b8[1, 512, 768]", sub_33: "f32[1, 512, 768]", sqrt_12: "f32[1, 512, 1]", view_72: "f32[512, 768]", convert_element_type_18: "b8[1, 12, 512, 512]", view_84: "f32[512, 768]", convert_element_type_19: "b8[1, 512, 768]", sub_38: "f32[1, 512, 768]", sqrt_14: "f32[1, 512, 1]", view_86: "f32[512, 768]", addmm_13: "f32[512, 3072]", view_88: "f32[512, 3072]", convert_element_type_20: "b8[1, 512, 768]", sub_41: "f32[1, 512, 768]", sqrt_15: "f32[1, 512, 1]", view_90: "f32[512, 768]", convert_element_type_22: "b8[1, 12, 512, 512]", view_102: "f32[512, 768]", convert_element_type_23: "b8[1, 512, 768]", sub_46: "f32[1, 512, 768]", sqrt_17: "f32[1, 512, 1]", view_104: "f32[512, 768]", addmm_16: "f32[512, 3072]", view_106: "f32[512, 3072]", convert_element_type_24: "b8[1, 512, 768]", sub_49: "f32[1, 512, 768]", sqrt_18: "f32[1, 512, 1]", view_108: "f32[512, 768]", convert_element_type_26: "b8[1, 12, 512, 512]", view_120: "f32[512, 768]", convert_element_type_27: "b8[1, 512, 768]", sub_54: "f32[1, 512, 768]", sqrt_20: "f32[1, 512, 1]", view_122: "f32[512, 768]", addmm_19: "f32[512, 3072]", view_124: "f32[512, 3072]", convert_element_type_28: "b8[1, 512, 768]", sub_57: "f32[1, 512, 768]", sqrt_21: "f32[1, 512, 1]", view_126: "f32[512, 768]", convert_element_type_30: "b8[1, 12, 512, 512]", view_138: "f32[512, 768]", convert_element_type_31: "b8[1, 512, 768]", sub_62: "f32[1, 512, 768]", sqrt_23: "f32[1, 512, 1]", view_140: "f32[512, 768]", addmm_22: "f32[512, 3072]", view_142: "f32[512, 3072]", convert_element_type_32: "b8[1, 512, 768]", sub_65: "f32[1, 512, 768]", sqrt_24: "f32[1, 512, 1]", view_144: "f32[512, 768]", convert_element_type_34: "b8[1, 12, 512, 512]", view_156: "f32[512, 768]", convert_element_type_35: "b8[1, 512, 768]", sub_70: "f32[1, 512, 768]", sqrt_26: "f32[1, 512, 1]", view_158: "f32[512, 768]", addmm_25: "f32[512, 3072]", view_160: "f32[512, 3072]", convert_element_type_36: "b8[1, 512, 768]", sub_73: "f32[1, 512, 768]", sqrt_27: "f32[1, 512, 1]", view_162: "f32[512, 768]", convert_element_type_38: "b8[1, 12, 512, 512]", view_174: "f32[512, 768]", convert_element_type_39: "b8[1, 512, 768]", sub_78: "f32[1, 512, 768]", sqrt_29: "f32[1, 512, 1]", view_176: "f32[512, 768]", addmm_28: "f32[512, 3072]", view_178: "f32[512, 3072]", convert_element_type_40: "b8[1, 512, 768]", sub_81: "f32[1, 512, 768]", sqrt_30: "f32[1, 512, 1]", view_180: "f32[512, 768]", convert_element_type_42: "b8[1, 12, 512, 512]", view_192: "f32[512, 768]", convert_element_type_43: "b8[1, 512, 768]", sub_86: "f32[1, 512, 768]", sqrt_32: "f32[1, 512, 1]", view_194: "f32[512, 768]", addmm_31: "f32[512, 3072]", view_196: "f32[512, 3072]", convert_element_type_44: "b8[1, 512, 768]", sub_89: "f32[1, 512, 768]", sqrt_33: "f32[1, 512, 1]", view_198: "f32[512, 768]", convert_element_type_46: "b8[1, 12, 512, 512]", view_210: "f32[512, 768]", convert_element_type_47: "b8[1, 512, 768]", sub_94: "f32[1, 512, 768]", sqrt_35: "f32[1, 512, 1]", view_212: "f32[512, 768]", addmm_34: "f32[512, 3072]", view_214: "f32[512, 3072]", convert_element_type_48: "b8[1, 512, 768]", sub_97: "f32[1, 512, 768]", sqrt_36: "f32[1, 512, 1]", view_216: "f32[512, 768]", addmm_36: "f32[512, 768]", mul_115: "f32[1, 512, 768]", view_218: "f32[512, 768]", sub_101: "f32[512, 50265]", convert_element_type_49: "f32[]", permute_147: "f32[50265, 768]", div_51: "f32[1, 512, 1]", permute_151: "f32[768, 768]", permute_155: "f32[768, 3072]", permute_159: "f32[3072, 768]", permute_163: "f32[768, 768]", permute_168: "f32[12, 512, 512]", permute_169: "f32[12, 64, 512]", alias_43: "f32[1, 12, 512, 512]", permute_170: "f32[12, 64, 512]", permute_171: "f32[12, 512, 64]", permute_178: "f32[2304, 768]", permute_180: "f32[768, 3072]", permute_184: "f32[3072, 768]", permute_188: "f32[768, 768]", permute_193: "f32[12, 512, 512]", permute_194: "f32[12, 64, 512]", alias_48: "f32[1, 12, 512, 512]", permute_195: "f32[12, 64, 512]", permute_196: "f32[12, 512, 64]", permute_203: "f32[2304, 768]", permute_205: "f32[768, 3072]", permute_209: "f32[3072, 768]", permute_213: "f32[768, 768]", permute_218: "f32[12, 512, 512]", permute_219: "f32[12, 64, 512]", alias_53: "f32[1, 12, 512, 512]", permute_220: "f32[12, 64, 512]", permute_221: "f32[12, 512, 64]", permute_228: "f32[2304, 768]", permute_230: "f32[768, 3072]", permute_234: "f32[3072, 768]", permute_238: "f32[768, 768]", permute_243: "f32[12, 512, 512]", permute_244: "f32[12, 64, 512]", alias_58: "f32[1, 12, 512, 512]", permute_245: "f32[12, 64, 512]", permute_246: "f32[12, 512, 64]", permute_253: "f32[2304, 768]", permute_255: "f32[768, 3072]", permute_259: "f32[3072, 768]", permute_263: "f32[768, 768]", permute_268: "f32[12, 512, 512]", permute_269: "f32[12, 64, 512]", alias_63: "f32[1, 12, 512, 512]", permute_270: "f32[12, 64, 512]", permute_271: "f32[12, 512, 64]", permute_278: "f32[2304, 768]", permute_280: "f32[768, 3072]", permute_284: "f32[3072, 768]", permute_288: "f32[768, 768]", permute_293: "f32[12, 512, 512]", permute_294: "f32[12, 64, 512]", alias_68: "f32[1, 12, 512, 512]", permute_295: "f32[12, 64, 512]", permute_296: "f32[12, 512, 64]", permute_303: "f32[2304, 768]", permute_305: "f32[768, 3072]", permute_309: "f32[3072, 768]", permute_313: "f32[768, 768]", permute_318: "f32[12, 512, 512]", permute_319: "f32[12, 64, 512]", alias_73: "f32[1, 12, 512, 512]", permute_320: "f32[12, 64, 512]", permute_321: "f32[12, 512, 64]", permute_328: "f32[2304, 768]", permute_330: "f32[768, 3072]", permute_334: "f32[3072, 768]", permute_338: "f32[768, 768]", permute_343: "f32[12, 512, 512]", permute_344: "f32[12, 64, 512]", alias_78: "f32[1, 12, 512, 512]", permute_345: "f32[12, 64, 512]", permute_346: "f32[12, 512, 64]", permute_353: "f32[2304, 768]", permute_355: "f32[768, 3072]", permute_359: "f32[3072, 768]", permute_363: "f32[768, 768]", permute_368: "f32[12, 512, 512]", permute_369: "f32[12, 64, 512]", alias_83: "f32[1, 12, 512, 512]", permute_370: "f32[12, 64, 512]", permute_371: "f32[12, 512, 64]", permute_378: "f32[2304, 768]", permute_380: "f32[768, 3072]", permute_384: "f32[3072, 768]", permute_388: "f32[768, 768]", permute_393: "f32[12, 512, 512]", permute_394: "f32[12, 64, 512]", alias_88: "f32[1, 12, 512, 512]", permute_395: "f32[12, 64, 512]", permute_396: "f32[12, 512, 64]", permute_403: "f32[2304, 768]", permute_405: "f32[768, 3072]", permute_409: "f32[3072, 768]", permute_413: "f32[768, 768]", permute_418: "f32[12, 512, 512]", permute_419: "f32[12, 64, 512]", alias_93: "f32[1, 12, 512, 512]", permute_420: "f32[12, 64, 512]", permute_421: "f32[12, 512, 64]", permute_428: "f32[2304, 768]", permute_430: "f32[768, 3072]", permute_434: "f32[3072, 768]", permute_438: "f32[768, 768]", permute_443: "f32[12, 512, 512]", permute_444: "f32[12, 64, 512]", alias_98: "f32[1, 12, 512, 512]", permute_445: "f32[12, 64, 512]", permute_446: "f32[12, 512, 64]", permute_453: "f32[2304, 768]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 50265]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt)
    div: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub, sqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:662, code: scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 8.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_2: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_2)
    div_3: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_6, sqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_15: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_1, [1, 512, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_9: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476)
    erf: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_8: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_3: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_3)
    div_4: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_9, sqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_5: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_5)
    div_7: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_14, sqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_33: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 512, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_18: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_1: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_17: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_6: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_6)
    div_8: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_17, sqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_8: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_8)
    div_11: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_22, sqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_51: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_7, [1, 512, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, 0.7071067811865476)
    erf_2: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_26: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_9: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_9)
    div_12: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_25, sqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_11: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_11)
    div_15: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_30, sqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_69: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 512, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_36: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, 0.7071067811865476)
    erf_3: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_35: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_12: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_12)
    div_16: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_33, sqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_14: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_14)
    div_19: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_38, sqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_13, [1, 512, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_45: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_4: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
    add_44: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_15: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_15)
    div_20: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_41, sqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_17: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_17)
    div_23: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_46, sqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_105: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 512, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, 0.7071067811865476)
    erf_5: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_53: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_18: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_18)
    div_24: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_49, sqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_20: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_20)
    div_27: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_54, sqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_123: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_19, [1, 512, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_63: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, 0.7071067811865476)
    erf_6: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_62: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_21: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_21)
    div_28: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_57, sqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_23: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_23)
    div_31: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_62, sqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_141: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 512, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_72: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, 0.7071067811865476)
    erf_7: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_72);  mul_72 = None
    add_71: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_24: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_24)
    div_32: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_65, sqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_26: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_26)
    div_35: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_70, sqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_159: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_25, [1, 512, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_81: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, 0.7071067811865476)
    erf_8: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_80: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_27: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_27)
    div_36: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_73, sqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_29: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_29)
    div_39: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_78, sqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_177: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 512, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_90: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476)
    erf_9: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
    add_89: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_30: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_30)
    div_40: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_81, sqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_32: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_32)
    div_43: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_86, sqrt_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_195: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_31, [1, 512, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_99: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_10: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_98: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_33: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_33)
    div_44: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_89, sqrt_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_35: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_35)
    div_47: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_94, sqrt_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_213: "f32[1, 512, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 512, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_108: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, 0.7071067811865476)
    erf_11: "f32[1, 512, 3072]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_107: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    alias_36: "f32[1, 512, 1]" = torch.ops.aten.alias.default(sqrt_36)
    div_48: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(sub_97, sqrt_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1116, code: hidden_states = self.dense(hidden_states)
    view_217: "f32[1, 512, 768]" = torch.ops.aten.view.default(addmm_36, [1, 512, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_113: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_12: "f32[1, 512, 768]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_111: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    view_221: "i64[512]" = torch.ops.aten.view.default(primals_169, [-1]);  primals_169 = None
    alias_37: "f32[512, 50265]" = torch.ops.aten.alias.default(sub_101);  sub_101 = None
    full_default_86: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_50: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_49);  tangents_1 = convert_element_type_49 = None
    unsqueeze_53: "i64[512, 1]" = torch.ops.aten.unsqueeze.default(view_221, 1);  view_221 = None
    ne_3: "b8[512, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_53, -100)
    where_63: "i64[512, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_53, full_default_86);  unsqueeze_53 = full_default_86 = None
    full_default_89: "f32[512, 50265]" = torch.ops.aten.full.default([512, 50265], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[512, 50265]" = torch.ops.aten.scatter.value(full_default_89, 1, where_63, -1.0);  full_default_89 = where_63 = None
    where_64: "f32[512, 1]" = torch.ops.aten.where.self(ne_3, div_50, full_default_1);  ne_3 = div_50 = None
    mul_117: "f32[512, 50265]" = torch.ops.aten.mul.Tensor(scatter, where_64);  scatter = where_64 = None
    alias_38: "f32[512, 50265]" = torch.ops.aten.alias.default(alias_37);  alias_37 = None
    exp_13: "f32[512, 50265]" = torch.ops.aten.exp.default(alias_38);  alias_38 = None
    sum_16: "f32[512, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [1], True)
    mul_118: "f32[512, 50265]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_102: "f32[512, 50265]" = torch.ops.aten.sub.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
    view_222: "f32[1, 512, 50265]" = torch.ops.aten.view.default(sub_102, [1, 512, 50265]);  sub_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1089, code: masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    add_114: "f32[1, 512, 50265]" = torch.ops.aten.add.Tensor(tangents_2, view_222);  tangents_2 = view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1139, code: hidden_states = self.decoder(hidden_states)
    view_223: "f32[512, 50265]" = torch.ops.aten.view.default(add_114, [512, 50265]);  add_114 = None
    mm_12: "f32[512, 768]" = torch.ops.aten.mm.default(view_223, permute_147);  permute_147 = None
    permute_148: "f32[50265, 512]" = torch.ops.aten.permute.default(view_223, [1, 0])
    mm_13: "f32[50265, 768]" = torch.ops.aten.mm.default(permute_148, view_218);  permute_148 = view_218 = None
    permute_149: "f32[768, 50265]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_17: "f32[1, 50265]" = torch.ops.aten.sum.dim_IntList(view_223, [0], True);  view_223 = None
    view_224: "f32[50265]" = torch.ops.aten.view.default(sum_17, [50265]);  sum_17 = None
    permute_150: "f32[50265, 768]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_225: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_12, [1, 512, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1118, code: hidden_states = self.LayerNorm(hidden_states)
    mul_120: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_225, primals_163);  primals_163 = None
    mul_121: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, 768)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
    mul_122: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_120, mul_115);  mul_120 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
    mul_123: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_115, sum_19);  sum_19 = None
    sub_104: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(mul_121, sum_18);  mul_121 = sum_18 = None
    sub_105: "f32[1, 512, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_123);  sub_104 = mul_123 = None
    mul_124: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_105);  div_51 = sub_105 = None
    mul_125: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_225, mul_115);  mul_115 = None
    sum_20: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_225, [0, 1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_127: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
    mul_128: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_129: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_128, -0.5);  mul_128 = None
    exp_14: "f32[1, 512, 768]" = torch.ops.aten.exp.default(mul_129);  mul_129 = None
    mul_130: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_131: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_130);  view_217 = mul_130 = None
    add_116: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(mul_127, mul_131);  mul_127 = mul_131 = None
    mul_132: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_124, add_116);  mul_124 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:1116, code: hidden_states = self.dense(hidden_states)
    view_226: "f32[512, 768]" = torch.ops.aten.view.default(mul_132, [512, 768]);  mul_132 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_226, permute_151);  permute_151 = None
    permute_152: "f32[768, 512]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_216);  permute_152 = view_216 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_228: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_14, [1, 512, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_23: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0, 1], True)
    view_229: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    mul_133: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, primals_73);  primals_73 = None
    mul_134: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(view_228, div_48);  view_228 = None
    sum_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1], True);  mul_134 = None
    view_230: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_53: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_48, sqrt_36);  div_48 = None
    neg_1: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_133)
    mul_135: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_1, div_53);  neg_1 = div_53 = None
    div_54: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_133, sqrt_36);  mul_133 = sqrt_36 = None
    sum_25: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    alias_39: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    mul_136: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_39, 2);  alias_39 = None
    div_55: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_25, mul_136);  sum_25 = mul_136 = None
    neg_2: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_54)
    sum_26: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_2, [2], True);  neg_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_48: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_55, [1, 512, 768]);  div_55 = None
    div_56: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
    pow_26: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_97, 1.0);  sub_97 = None
    mul_137: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_26, 2.0);  pow_26 = None
    mul_138: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_56, mul_137);  div_56 = mul_137 = None
    neg_3: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_138)
    sum_27: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_3, [2], True);  neg_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_117: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_54, mul_138);  div_54 = mul_138 = None
    add_118: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_26, sum_27);  sum_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_49: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_118, [1, 512, 768]);  add_118 = None
    div_57: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_119: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_117, div_57);  add_117 = div_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_65: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_48, full_default_1, add_119);  convert_element_type_48 = None
    mul_139: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_65, 1.1111111111111112);  where_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_231: "f32[512, 768]" = torch.ops.aten.view.default(mul_139, [512, 768]);  mul_139 = None
    mm_16: "f32[512, 3072]" = torch.ops.aten.mm.default(view_231, permute_155);  permute_155 = None
    permute_156: "f32[768, 512]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_17: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_156, view_214);  permute_156 = view_214 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
    permute_158: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_233: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_16, [1, 512, 3072]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_141: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.5);  add_107 = None
    mul_142: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, view_213)
    mul_143: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_142, -0.5);  mul_142 = None
    exp_15: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_143);  mul_143 = None
    mul_144: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_145: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_213, mul_144);  view_213 = mul_144 = None
    add_121: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_141, mul_145);  mul_141 = mul_145 = None
    mul_146: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_233, add_121);  view_233 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_234: "f32[512, 3072]" = torch.ops.aten.view.default(mul_146, [512, 3072]);  mul_146 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_234, permute_159);  permute_159 = None
    permute_160: "f32[3072, 512]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_19: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_160, view_212);  permute_160 = view_212 = None
    permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[3072]" = torch.ops.aten.view.default(sum_29, [3072]);  sum_29 = None
    permute_162: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_236: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_18, [1, 512, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_122: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_119, view_236);  add_119 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1], True)
    view_237: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    mul_147: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, primals_71);  primals_71 = None
    mul_148: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_122, div_47);  add_122 = None
    sum_31: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1], True);  mul_148 = None
    view_238: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_59: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_47, sqrt_35);  div_47 = None
    neg_4: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_147)
    mul_149: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_4, div_59);  neg_4 = div_59 = None
    div_60: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_147, sqrt_35);  mul_147 = sqrt_35 = None
    sum_32: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_149, [2], True);  mul_149 = None
    alias_40: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_150: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_40, 2);  alias_40 = None
    div_61: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_32, mul_150);  sum_32 = mul_150 = None
    neg_5: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_60)
    sum_33: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_5, [2], True);  neg_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_50: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_61, [1, 512, 768]);  div_61 = None
    div_62: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
    pow_27: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_94, 1.0);  sub_94 = None
    mul_151: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_27, 2.0);  pow_27 = None
    mul_152: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_62, mul_151);  div_62 = mul_151 = None
    neg_6: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_152)
    sum_34: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_6, [2], True);  neg_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_123: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_60, mul_152);  div_60 = mul_152 = None
    add_124: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_33, sum_34);  sum_33 = sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_51: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_124, [1, 512, 768]);  add_124 = None
    div_63: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_125: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_123, div_63);  add_123 = div_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_66: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_47, full_default_1, add_125);  convert_element_type_47 = None
    mul_153: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_66, 1.1111111111111112);  where_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_239: "f32[512, 768]" = torch.ops.aten.view.default(mul_153, [512, 768]);  mul_153 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_239, permute_163);  permute_163 = None
    permute_164: "f32[768, 512]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_164, view_210);  permute_164 = view_210 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_241: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_20, [1, 512, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_242: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_241, [1, 512, 12, 64]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_167: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_242, [0, 2, 1, 3]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_243: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_167, [12, 512, 64]);  permute_167 = None
    bmm_24: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_168, view_243);  permute_168 = None
    bmm_25: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_243, permute_169);  view_243 = permute_169 = None
    view_244: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_24, [1, 12, 512, 64]);  bmm_24 = None
    view_245: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_25, [1, 12, 512, 512]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_67: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_46, full_default_1, view_245);  convert_element_type_46 = view_245 = None
    mul_154: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_67, 1.1111111111111112);  where_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_155: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_154, alias_43);  mul_154 = None
    sum_36: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [-1], True)
    mul_156: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_43, sum_36);  alias_43 = sum_36 = None
    sub_106: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_155, mul_156);  mul_155 = mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_246: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_106, [12, 512, 512]);  sub_106 = None
    bmm_26: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_170, view_246);  permute_170 = None
    bmm_27: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_246, permute_171);  view_246 = permute_171 = None
    view_247: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_26, [1, 12, 64, 512]);  bmm_26 = None
    view_248: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_27, [1, 12, 512, 64]);  bmm_27 = None
    permute_172: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_247, [0, 1, 3, 2]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_64: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_248, full_default_2);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_37: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_244, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_173: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_37, [0, 2, 1, 3]);  sum_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_249: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_173, [1, 1, 768]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    full_default_94: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_249, 2, 0, 9223372036854775807);  view_249 = None
    squeeze_2: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter, 1);  slice_scatter = None
    squeeze_3: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_2, 0);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_38: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_64, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_174: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_38, [0, 2, 1, 3]);  sum_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_250: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_174, [1, 1, 768]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_1: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_250, 2, 0, 9223372036854775807);  view_250 = None
    squeeze_4: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_1, 1);  slice_scatter_1 = None
    squeeze_5: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_4, 0);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_64, permute_172, view_244], 3);  div_64 = permute_172 = view_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_175: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat, [0, 2, 1, 3]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_12: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_251: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_12, [1, 512, 2304]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_252: "f32[512, 2304]" = torch.ops.aten.view.default(view_251, [512, 2304]);  view_251 = None
    permute_176: "f32[2304, 512]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_22: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_176, view_198);  permute_176 = view_198 = None
    permute_177: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_22, [1, 0]);  mm_22 = None
    mm_23: "f32[512, 768]" = torch.ops.aten.mm.default(view_252, permute_178);  view_252 = permute_178 = None
    view_253: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_23, [1, 512, 768]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_126: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_125, view_253);  add_125 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_179: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1], True)
    view_254: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    mul_157: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, primals_67);  primals_67 = None
    mul_158: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_126, div_44);  add_126 = None
    sum_40: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_255: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_66: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_44, sqrt_33);  div_44 = None
    neg_7: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_157)
    mul_159: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_7, div_66);  neg_7 = div_66 = None
    div_67: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_157, sqrt_33);  mul_157 = sqrt_33 = None
    sum_41: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    alias_44: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_160: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_44, 2);  alias_44 = None
    div_68: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_41, mul_160);  sum_41 = mul_160 = None
    neg_8: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_67)
    sum_42: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_8, [2], True);  neg_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_52: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_68, [1, 512, 768]);  div_68 = None
    div_69: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
    pow_28: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_89, 1.0);  sub_89 = None
    mul_161: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_28, 2.0);  pow_28 = None
    mul_162: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_69, mul_161);  div_69 = mul_161 = None
    neg_9: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_162)
    sum_43: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True);  neg_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_127: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_67, mul_162);  div_67 = mul_162 = None
    add_128: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_42, sum_43);  sum_42 = sum_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_53: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_128, [1, 512, 768]);  add_128 = None
    div_70: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_129: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_127, div_70);  add_127 = div_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_68: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_44, full_default_1, add_129);  convert_element_type_44 = None
    mul_163: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_68, 1.1111111111111112);  where_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_256: "f32[512, 768]" = torch.ops.aten.view.default(mul_163, [512, 768]);  mul_163 = None
    mm_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_256, permute_180);  permute_180 = None
    permute_181: "f32[768, 512]" = torch.ops.aten.permute.default(view_256, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_181, view_196);  permute_181 = view_196 = None
    permute_182: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True);  view_256 = None
    view_257: "f32[768]" = torch.ops.aten.view.default(sum_44, [768]);  sum_44 = None
    permute_183: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_258: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_24, [1, 512, 3072]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_165: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_166: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_167: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_166, -0.5);  mul_166 = None
    exp_16: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_167);  mul_167 = None
    mul_168: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_169: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_168);  view_195 = mul_168 = None
    add_131: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_165, mul_169);  mul_165 = mul_169 = None
    mul_170: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_258, add_131);  view_258 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_259: "f32[512, 3072]" = torch.ops.aten.view.default(mul_170, [512, 3072]);  mul_170 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_259, permute_184);  permute_184 = None
    permute_185: "f32[3072, 512]" = torch.ops.aten.permute.default(view_259, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_185, view_194);  permute_185 = view_194 = None
    permute_186: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_45: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True);  view_259 = None
    view_260: "f32[3072]" = torch.ops.aten.view.default(sum_45, [3072]);  sum_45 = None
    permute_187: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_261: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_26, [1, 512, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_132: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_129, view_261);  add_129 = view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_46: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1], True)
    view_262: "f32[768]" = torch.ops.aten.view.default(sum_46, [768]);  sum_46 = None
    mul_171: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_65);  primals_65 = None
    mul_172: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_132, div_43);  add_132 = None
    sum_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1], True);  mul_172 = None
    view_263: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_72: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_43, sqrt_32);  div_43 = None
    neg_10: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_171)
    mul_173: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_10, div_72);  neg_10 = div_72 = None
    div_73: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_171, sqrt_32);  mul_171 = sqrt_32 = None
    sum_48: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    alias_45: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_174: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_45, 2);  alias_45 = None
    div_74: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_48, mul_174);  sum_48 = mul_174 = None
    neg_11: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_73)
    sum_49: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True);  neg_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_54: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_74, [1, 512, 768]);  div_74 = None
    div_75: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
    pow_29: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_86, 1.0);  sub_86 = None
    mul_175: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_29, 2.0);  pow_29 = None
    mul_176: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_75, mul_175);  div_75 = mul_175 = None
    neg_12: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_176)
    sum_50: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_12, [2], True);  neg_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_133: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_73, mul_176);  div_73 = mul_176 = None
    add_134: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_49, sum_50);  sum_49 = sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_55: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_134, [1, 512, 768]);  add_134 = None
    div_76: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_135: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_133, div_76);  add_133 = div_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_69: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_43, full_default_1, add_135);  convert_element_type_43 = None
    mul_177: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_69, 1.1111111111111112);  where_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(mul_177, [512, 768]);  mul_177 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_264, permute_188);  permute_188 = None
    permute_189: "f32[768, 512]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_29: "f32[768, 768]" = torch.ops.aten.mm.default(permute_189, view_192);  permute_189 = view_192 = None
    permute_190: "f32[768, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    view_266: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_28, [1, 512, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_267: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_266, [1, 512, 12, 64]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_192: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_268: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_192, [12, 512, 64]);  permute_192 = None
    bmm_28: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_193, view_268);  permute_193 = None
    bmm_29: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_268, permute_194);  view_268 = permute_194 = None
    view_269: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_28, [1, 12, 512, 64]);  bmm_28 = None
    view_270: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_29, [1, 12, 512, 512]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_70: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_42, full_default_1, view_270);  convert_element_type_42 = view_270 = None
    mul_178: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_70, 1.1111111111111112);  where_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_179: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_178, alias_48);  mul_178 = None
    sum_52: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [-1], True)
    mul_180: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_48, sum_52);  alias_48 = sum_52 = None
    sub_107: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_271: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_107, [12, 512, 512]);  sub_107 = None
    bmm_30: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_195, view_271);  permute_195 = None
    bmm_31: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_271, permute_196);  view_271 = permute_196 = None
    view_272: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_30, [1, 12, 64, 512]);  bmm_30 = None
    view_273: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_31, [1, 12, 512, 64]);  bmm_31 = None
    permute_197: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 2]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_77: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_273, full_default_2);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_53: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_269, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_198: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_53, [0, 2, 1, 3]);  sum_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_274: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_198, [1, 1, 768]);  permute_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_2: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_274, 2, 0, 9223372036854775807);  view_274 = None
    squeeze_6: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_2, 1);  slice_scatter_2 = None
    squeeze_7: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_6, 0);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_54: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_77, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_199: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_54, [0, 2, 1, 3]);  sum_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_275: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_199, [1, 1, 768]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_3: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_275, 2, 0, 9223372036854775807);  view_275 = None
    squeeze_8: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_3, 1);  slice_scatter_3 = None
    squeeze_9: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_8, 0);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_1: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_77, permute_197, view_269], 3);  div_77 = permute_197 = view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_1, [0, 2, 1, 3]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_13: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    view_276: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_13, [1, 512, 2304]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_277: "f32[512, 2304]" = torch.ops.aten.view.default(view_276, [512, 2304]);  view_276 = None
    permute_201: "f32[2304, 512]" = torch.ops.aten.permute.default(view_277, [1, 0])
    mm_30: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_201, view_180);  permute_201 = view_180 = None
    permute_202: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_30, [1, 0]);  mm_30 = None
    mm_31: "f32[512, 768]" = torch.ops.aten.mm.default(view_277, permute_203);  view_277 = permute_203 = None
    view_278: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_31, [1, 512, 768]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_136: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_135, view_278);  add_135 = view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_204: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_55: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_136, [0, 1], True)
    view_279: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    mul_181: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_136, primals_61);  primals_61 = None
    mul_182: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_136, div_40);  add_136 = None
    sum_56: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1], True);  mul_182 = None
    view_280: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_79: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_40, sqrt_30);  div_40 = None
    neg_13: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_181)
    mul_183: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_13, div_79);  neg_13 = div_79 = None
    div_80: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_181, sqrt_30);  mul_181 = sqrt_30 = None
    sum_57: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
    alias_49: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    mul_184: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_49, 2);  alias_49 = None
    div_81: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_57, mul_184);  sum_57 = mul_184 = None
    neg_14: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_80)
    sum_58: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_14, [2], True);  neg_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_56: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_81, [1, 512, 768]);  div_81 = None
    div_82: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
    pow_30: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_81, 1.0);  sub_81 = None
    mul_185: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_30, 2.0);  pow_30 = None
    mul_186: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_82, mul_185);  div_82 = mul_185 = None
    neg_15: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_186)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_15, [2], True);  neg_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_137: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_80, mul_186);  div_80 = mul_186 = None
    add_138: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_58, sum_59);  sum_58 = sum_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_57: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_138, [1, 512, 768]);  add_138 = None
    div_83: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_139: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_137, div_83);  add_137 = div_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_71: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_40, full_default_1, add_139);  convert_element_type_40 = None
    mul_187: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_71, 1.1111111111111112);  where_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_281: "f32[512, 768]" = torch.ops.aten.view.default(mul_187, [512, 768]);  mul_187 = None
    mm_32: "f32[512, 3072]" = torch.ops.aten.mm.default(view_281, permute_205);  permute_205 = None
    permute_206: "f32[768, 512]" = torch.ops.aten.permute.default(view_281, [1, 0])
    mm_33: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_206, view_178);  permute_206 = view_178 = None
    permute_207: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_281, [0], True);  view_281 = None
    view_282: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    permute_208: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_283: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_32, [1, 512, 3072]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_189: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_190: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, view_177)
    mul_191: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_190, -0.5);  mul_190 = None
    exp_17: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_191);  mul_191 = None
    mul_192: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_193: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_177, mul_192);  view_177 = mul_192 = None
    add_141: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_189, mul_193);  mul_189 = mul_193 = None
    mul_194: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_283, add_141);  view_283 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_284: "f32[512, 3072]" = torch.ops.aten.view.default(mul_194, [512, 3072]);  mul_194 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_284, permute_209);  permute_209 = None
    permute_210: "f32[3072, 512]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_35: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_210, view_176);  permute_210 = view_176 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_61: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[3072]" = torch.ops.aten.view.default(sum_61, [3072]);  sum_61 = None
    permute_212: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_286: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_34, [1, 512, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_142: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_139, view_286);  add_139 = view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1], True)
    view_287: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    mul_195: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_59);  primals_59 = None
    mul_196: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_142, div_39);  add_142 = None
    sum_63: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_196, [0, 1], True);  mul_196 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_63, [768]);  sum_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_85: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_39, sqrt_29);  div_39 = None
    neg_16: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_195)
    mul_197: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_16, div_85);  neg_16 = div_85 = None
    div_86: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_195, sqrt_29);  mul_195 = sqrt_29 = None
    sum_64: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
    alias_50: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_198: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_50, 2);  alias_50 = None
    div_87: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_64, mul_198);  sum_64 = mul_198 = None
    neg_17: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_86)
    sum_65: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_17, [2], True);  neg_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_58: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_87, [1, 512, 768]);  div_87 = None
    div_88: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
    pow_31: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_78, 1.0);  sub_78 = None
    mul_199: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_31, 2.0);  pow_31 = None
    mul_200: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_88, mul_199);  div_88 = mul_199 = None
    neg_18: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_200)
    sum_66: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_18, [2], True);  neg_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_143: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_86, mul_200);  div_86 = mul_200 = None
    add_144: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_65, sum_66);  sum_65 = sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_59: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_144, [1, 512, 768]);  add_144 = None
    div_89: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_145: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_143, div_89);  add_143 = div_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_72: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_39, full_default_1, add_145);  convert_element_type_39 = None
    mul_201: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_72, 1.1111111111111112);  where_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_289: "f32[512, 768]" = torch.ops.aten.view.default(mul_201, [512, 768]);  mul_201 = None
    mm_36: "f32[512, 768]" = torch.ops.aten.mm.default(view_289, permute_213);  permute_213 = None
    permute_214: "f32[768, 512]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_37: "f32[768, 768]" = torch.ops.aten.mm.default(permute_214, view_174);  permute_214 = view_174 = None
    permute_215: "f32[768, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_216: "f32[768, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_291: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_36, [1, 512, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_292: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_291, [1, 512, 12, 64]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_217: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_293: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_217, [12, 512, 64]);  permute_217 = None
    bmm_32: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_218, view_293);  permute_218 = None
    bmm_33: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_293, permute_219);  view_293 = permute_219 = None
    view_294: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_32, [1, 12, 512, 64]);  bmm_32 = None
    view_295: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_33, [1, 12, 512, 512]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_73: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_38, full_default_1, view_295);  convert_element_type_38 = view_295 = None
    mul_202: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_73, 1.1111111111111112);  where_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_203: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_202, alias_53);  mul_202 = None
    sum_68: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [-1], True)
    mul_204: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_53, sum_68);  alias_53 = sum_68 = None
    sub_108: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_203, mul_204);  mul_203 = mul_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_296: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_108, [12, 512, 512]);  sub_108 = None
    bmm_34: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_220, view_296);  permute_220 = None
    bmm_35: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_296, permute_221);  view_296 = permute_221 = None
    view_297: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_34, [1, 12, 64, 512]);  bmm_34 = None
    view_298: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_35, [1, 12, 512, 64]);  bmm_35 = None
    permute_222: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_90: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_298, full_default_2);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_69: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_294, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_223: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_69, [0, 2, 1, 3]);  sum_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_299: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_223, [1, 1, 768]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_4: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_299, 2, 0, 9223372036854775807);  view_299 = None
    squeeze_10: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_4, 1);  slice_scatter_4 = None
    squeeze_11: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_10, 0);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_70: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_90, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_224: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_70, [0, 2, 1, 3]);  sum_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_300: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_224, [1, 1, 768]);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_5: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_300, 2, 0, 9223372036854775807);  view_300 = None
    squeeze_12: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_5, 1);  slice_scatter_5 = None
    squeeze_13: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_12, 0);  squeeze_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_2: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_90, permute_222, view_294], 3);  div_90 = permute_222 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_225: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_14: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    view_301: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_14, [1, 512, 2304]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_302: "f32[512, 2304]" = torch.ops.aten.view.default(view_301, [512, 2304]);  view_301 = None
    permute_226: "f32[2304, 512]" = torch.ops.aten.permute.default(view_302, [1, 0])
    mm_38: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_226, view_162);  permute_226 = view_162 = None
    permute_227: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_38, [1, 0]);  mm_38 = None
    mm_39: "f32[512, 768]" = torch.ops.aten.mm.default(view_302, permute_228);  view_302 = permute_228 = None
    view_303: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_39, [1, 512, 768]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_146: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_145, view_303);  add_145 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_229: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_146, [0, 1], True)
    view_304: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    mul_205: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, primals_55);  primals_55 = None
    mul_206: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_146, div_36);  add_146 = None
    sum_72: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1], True);  mul_206 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_92: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_36, sqrt_27);  div_36 = None
    neg_19: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_205)
    mul_207: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_19, div_92);  neg_19 = div_92 = None
    div_93: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_205, sqrt_27);  mul_205 = sqrt_27 = None
    sum_73: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True);  mul_207 = None
    alias_54: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_208: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_54, 2);  alias_54 = None
    div_94: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_73, mul_208);  sum_73 = mul_208 = None
    neg_20: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_93)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_20, [2], True);  neg_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_60: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_94, [1, 512, 768]);  div_94 = None
    div_95: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
    pow_32: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_73, 1.0);  sub_73 = None
    mul_209: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_32, 2.0);  pow_32 = None
    mul_210: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_95, mul_209);  div_95 = mul_209 = None
    neg_21: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_210)
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True);  neg_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_147: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_93, mul_210);  div_93 = mul_210 = None
    add_148: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_74, sum_75);  sum_74 = sum_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_61: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_148, [1, 512, 768]);  add_148 = None
    div_96: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_149: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_147, div_96);  add_147 = div_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_74: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_36, full_default_1, add_149);  convert_element_type_36 = None
    mul_211: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_74, 1.1111111111111112);  where_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_306: "f32[512, 768]" = torch.ops.aten.view.default(mul_211, [512, 768]);  mul_211 = None
    mm_40: "f32[512, 3072]" = torch.ops.aten.mm.default(view_306, permute_230);  permute_230 = None
    permute_231: "f32[768, 512]" = torch.ops.aten.permute.default(view_306, [1, 0])
    mm_41: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_231, view_160);  permute_231 = view_160 = None
    permute_232: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_306, [0], True);  view_306 = None
    view_307: "f32[768]" = torch.ops.aten.view.default(sum_76, [768]);  sum_76 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_308: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_40, [1, 512, 3072]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_213: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_214: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, view_159)
    mul_215: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_214, -0.5);  mul_214 = None
    exp_18: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_215);  mul_215 = None
    mul_216: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_217: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_159, mul_216);  view_159 = mul_216 = None
    add_151: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_213, mul_217);  mul_213 = mul_217 = None
    mul_218: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_308, add_151);  view_308 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_309: "f32[512, 3072]" = torch.ops.aten.view.default(mul_218, [512, 3072]);  mul_218 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_309, permute_234);  permute_234 = None
    permute_235: "f32[3072, 512]" = torch.ops.aten.permute.default(view_309, [1, 0])
    mm_43: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_235, view_158);  permute_235 = view_158 = None
    permute_236: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_77: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_309, [0], True);  view_309 = None
    view_310: "f32[3072]" = torch.ops.aten.view.default(sum_77, [3072]);  sum_77 = None
    permute_237: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_311: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_42, [1, 512, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_152: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_149, view_311);  add_149 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_78: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1], True)
    view_312: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    mul_219: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, primals_53);  primals_53 = None
    mul_220: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_152, div_35);  add_152 = None
    sum_79: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True);  mul_220 = None
    view_313: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_98: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_35, sqrt_26);  div_35 = None
    neg_22: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_219)
    mul_221: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_22, div_98);  neg_22 = div_98 = None
    div_99: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_219, sqrt_26);  mul_219 = sqrt_26 = None
    sum_80: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_221, [2], True);  mul_221 = None
    alias_55: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_222: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_55, 2);  alias_55 = None
    div_100: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_80, mul_222);  sum_80 = mul_222 = None
    neg_23: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_99)
    sum_81: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_23, [2], True);  neg_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_62: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_100, [1, 512, 768]);  div_100 = None
    div_101: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
    pow_33: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_70, 1.0);  sub_70 = None
    mul_223: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_33, 2.0);  pow_33 = None
    mul_224: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_101, mul_223);  div_101 = mul_223 = None
    neg_24: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_224)
    sum_82: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_24, [2], True);  neg_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_153: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_99, mul_224);  div_99 = mul_224 = None
    add_154: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_81, sum_82);  sum_81 = sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_63: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_154, [1, 512, 768]);  add_154 = None
    div_102: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_155: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_153, div_102);  add_153 = div_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_75: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_35, full_default_1, add_155);  convert_element_type_35 = None
    mul_225: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_75, 1.1111111111111112);  where_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_314: "f32[512, 768]" = torch.ops.aten.view.default(mul_225, [512, 768]);  mul_225 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_314, permute_238);  permute_238 = None
    permute_239: "f32[768, 512]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_239, view_156);  permute_239 = view_156 = None
    permute_240: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_316: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_44, [1, 512, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_317: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_316, [1, 512, 12, 64]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_242: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_318: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_242, [12, 512, 64]);  permute_242 = None
    bmm_36: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_243, view_318);  permute_243 = None
    bmm_37: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_318, permute_244);  view_318 = permute_244 = None
    view_319: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_36, [1, 12, 512, 64]);  bmm_36 = None
    view_320: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_37, [1, 12, 512, 512]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_76: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_34, full_default_1, view_320);  convert_element_type_34 = view_320 = None
    mul_226: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_76, 1.1111111111111112);  where_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_227: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_226, alias_58);  mul_226 = None
    sum_84: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [-1], True)
    mul_228: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_58, sum_84);  alias_58 = sum_84 = None
    sub_109: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_321: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_109, [12, 512, 512]);  sub_109 = None
    bmm_38: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_245, view_321);  permute_245 = None
    bmm_39: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_321, permute_246);  view_321 = permute_246 = None
    view_322: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_38, [1, 12, 64, 512]);  bmm_38 = None
    view_323: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_39, [1, 12, 512, 64]);  bmm_39 = None
    permute_247: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_322, [0, 1, 3, 2]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_103: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_323, full_default_2);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_85: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_319, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_248: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_85, [0, 2, 1, 3]);  sum_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_324: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_248, [1, 1, 768]);  permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_6: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_324, 2, 0, 9223372036854775807);  view_324 = None
    squeeze_14: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_6, 1);  slice_scatter_6 = None
    squeeze_15: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_14, 0);  squeeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_86: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_103, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_249: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_86, [0, 2, 1, 3]);  sum_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_325: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_249, [1, 1, 768]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_7: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_325, 2, 0, 9223372036854775807);  view_325 = None
    squeeze_16: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_7, 1);  slice_scatter_7 = None
    squeeze_17: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_16, 0);  squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_3: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_103, permute_247, view_319], 3);  div_103 = permute_247 = view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_250: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
    view_326: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_15, [1, 512, 2304]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_327: "f32[512, 2304]" = torch.ops.aten.view.default(view_326, [512, 2304]);  view_326 = None
    permute_251: "f32[2304, 512]" = torch.ops.aten.permute.default(view_327, [1, 0])
    mm_46: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_251, view_144);  permute_251 = view_144 = None
    permute_252: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_46, [1, 0]);  mm_46 = None
    mm_47: "f32[512, 768]" = torch.ops.aten.mm.default(view_327, permute_253);  view_327 = permute_253 = None
    view_328: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_47, [1, 512, 768]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_156: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_155, view_328);  add_155 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_254: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_87: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1], True)
    view_329: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    mul_229: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_49);  primals_49 = None
    mul_230: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_156, div_32);  add_156 = None
    sum_88: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1], True);  mul_230 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_105: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_32, sqrt_24);  div_32 = None
    neg_25: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_229)
    mul_231: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_25, div_105);  neg_25 = div_105 = None
    div_106: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_229, sqrt_24);  mul_229 = sqrt_24 = None
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
    alias_59: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_232: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_59, 2);  alias_59 = None
    div_107: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_89, mul_232);  sum_89 = mul_232 = None
    neg_26: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_106)
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_26, [2], True);  neg_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_64: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_107, [1, 512, 768]);  div_107 = None
    div_108: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
    pow_34: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_65, 1.0);  sub_65 = None
    mul_233: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
    mul_234: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_108, mul_233);  div_108 = mul_233 = None
    neg_27: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_234)
    sum_91: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_27, [2], True);  neg_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_157: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_106, mul_234);  div_106 = mul_234 = None
    add_158: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_90, sum_91);  sum_90 = sum_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_65: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_158, [1, 512, 768]);  add_158 = None
    div_109: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_159: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_157, div_109);  add_157 = div_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_77: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_32, full_default_1, add_159);  convert_element_type_32 = None
    mul_235: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_77, 1.1111111111111112);  where_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_331: "f32[512, 768]" = torch.ops.aten.view.default(mul_235, [512, 768]);  mul_235 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_331, permute_255);  permute_255 = None
    permute_256: "f32[768, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_256, view_142);  permute_256 = view_142 = None
    permute_257: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    permute_258: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_333: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_48, [1, 512, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_237: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
    mul_238: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, view_141)
    mul_239: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_238, -0.5);  mul_238 = None
    exp_19: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_239);  mul_239 = None
    mul_240: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_241: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_141, mul_240);  view_141 = mul_240 = None
    add_161: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_237, mul_241);  mul_237 = mul_241 = None
    mul_242: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_333, add_161);  view_333 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_334: "f32[512, 3072]" = torch.ops.aten.view.default(mul_242, [512, 3072]);  mul_242 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_334, permute_259);  permute_259 = None
    permute_260: "f32[3072, 512]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_260, view_140);  permute_260 = view_140 = None
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[3072]" = torch.ops.aten.view.default(sum_93, [3072]);  sum_93 = None
    permute_262: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    view_336: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_50, [1, 512, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_162: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_159, view_336);  add_159 = view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1], True)
    view_337: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    mul_243: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, primals_47);  primals_47 = None
    mul_244: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_162, div_31);  add_162 = None
    sum_95: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_244, [0, 1], True);  mul_244 = None
    view_338: "f32[768]" = torch.ops.aten.view.default(sum_95, [768]);  sum_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_111: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_31, sqrt_23);  div_31 = None
    neg_28: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_243)
    mul_245: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_28, div_111);  neg_28 = div_111 = None
    div_112: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_243, sqrt_23);  mul_243 = sqrt_23 = None
    sum_96: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
    alias_60: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_246: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_60, 2);  alias_60 = None
    div_113: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_96, mul_246);  sum_96 = mul_246 = None
    neg_29: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_112)
    sum_97: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True);  neg_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_66: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_113, [1, 512, 768]);  div_113 = None
    div_114: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
    pow_35: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_62, 1.0);  sub_62 = None
    mul_247: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_35, 2.0);  pow_35 = None
    mul_248: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_114, mul_247);  div_114 = mul_247 = None
    neg_30: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_248)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_30, [2], True);  neg_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_163: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_112, mul_248);  div_112 = mul_248 = None
    add_164: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_97, sum_98);  sum_97 = sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_67: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_164, [1, 512, 768]);  add_164 = None
    div_115: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_165: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_163, div_115);  add_163 = div_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_78: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_31, full_default_1, add_165);  convert_element_type_31 = None
    mul_249: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_78, 1.1111111111111112);  where_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_339: "f32[512, 768]" = torch.ops.aten.view.default(mul_249, [512, 768]);  mul_249 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_339, permute_263);  permute_263 = None
    permute_264: "f32[768, 512]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_264, view_138);  permute_264 = view_138 = None
    permute_265: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_341: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_52, [1, 512, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_342: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_341, [1, 512, 12, 64]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_267: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_343: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_267, [12, 512, 64]);  permute_267 = None
    bmm_40: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_268, view_343);  permute_268 = None
    bmm_41: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_343, permute_269);  view_343 = permute_269 = None
    view_344: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_40, [1, 12, 512, 64]);  bmm_40 = None
    view_345: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_41, [1, 12, 512, 512]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_79: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_30, full_default_1, view_345);  convert_element_type_30 = view_345 = None
    mul_250: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_79, 1.1111111111111112);  where_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_251: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_250, alias_63);  mul_250 = None
    sum_100: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
    mul_252: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_63, sum_100);  alias_63 = sum_100 = None
    sub_110: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_251, mul_252);  mul_251 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_346: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_110, [12, 512, 512]);  sub_110 = None
    bmm_42: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_270, view_346);  permute_270 = None
    bmm_43: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_346, permute_271);  view_346 = permute_271 = None
    view_347: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_42, [1, 12, 64, 512]);  bmm_42 = None
    view_348: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_43, [1, 12, 512, 64]);  bmm_43 = None
    permute_272: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_347, [0, 1, 3, 2]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_116: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_348, full_default_2);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_101: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_344, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_273: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_101, [0, 2, 1, 3]);  sum_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_349: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_273, [1, 1, 768]);  permute_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_8: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_349, 2, 0, 9223372036854775807);  view_349 = None
    squeeze_18: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_8, 1);  slice_scatter_8 = None
    squeeze_19: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_18, 0);  squeeze_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_102: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_116, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_274: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_102, [0, 2, 1, 3]);  sum_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_350: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_274, [1, 1, 768]);  permute_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_9: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_350, 2, 0, 9223372036854775807);  view_350 = None
    squeeze_20: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_9, 1);  slice_scatter_9 = None
    squeeze_21: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_20, 0);  squeeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_4: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_116, permute_272, view_344], 3);  div_116 = permute_272 = view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_275: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_4, [0, 2, 1, 3]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_275, memory_format = torch.contiguous_format);  permute_275 = None
    view_351: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_16, [1, 512, 2304]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_352: "f32[512, 2304]" = torch.ops.aten.view.default(view_351, [512, 2304]);  view_351 = None
    permute_276: "f32[2304, 512]" = torch.ops.aten.permute.default(view_352, [1, 0])
    mm_54: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_276, view_126);  permute_276 = view_126 = None
    permute_277: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_54, [1, 0]);  mm_54 = None
    mm_55: "f32[512, 768]" = torch.ops.aten.mm.default(view_352, permute_278);  view_352 = permute_278 = None
    view_353: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_55, [1, 512, 768]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_166: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_165, view_353);  add_165 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_279: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_103: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_166, [0, 1], True)
    view_354: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    mul_253: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_166, primals_43);  primals_43 = None
    mul_254: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_166, div_28);  add_166 = None
    sum_104: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1], True);  mul_254 = None
    view_355: "f32[768]" = torch.ops.aten.view.default(sum_104, [768]);  sum_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_118: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_28, sqrt_21);  div_28 = None
    neg_31: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_253)
    mul_255: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_31, div_118);  neg_31 = div_118 = None
    div_119: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_253, sqrt_21);  mul_253 = sqrt_21 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    alias_64: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_256: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_64, 2);  alias_64 = None
    div_120: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_105, mul_256);  sum_105 = mul_256 = None
    neg_32: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_119)
    sum_106: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_32, [2], True);  neg_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_68: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_120, [1, 512, 768]);  div_120 = None
    div_121: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
    pow_36: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_57, 1.0);  sub_57 = None
    mul_257: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
    mul_258: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_121, mul_257);  div_121 = mul_257 = None
    neg_33: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_258)
    sum_107: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_33, [2], True);  neg_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_167: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_119, mul_258);  div_119 = mul_258 = None
    add_168: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_106, sum_107);  sum_106 = sum_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_69: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_168, [1, 512, 768]);  add_168 = None
    div_122: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_169: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_167, div_122);  add_167 = div_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_80: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_28, full_default_1, add_169);  convert_element_type_28 = None
    mul_259: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_80, 1.1111111111111112);  where_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_356: "f32[512, 768]" = torch.ops.aten.view.default(mul_259, [512, 768]);  mul_259 = None
    mm_56: "f32[512, 3072]" = torch.ops.aten.mm.default(view_356, permute_280);  permute_280 = None
    permute_281: "f32[768, 512]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_57: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_281, view_124);  permute_281 = view_124 = None
    permute_282: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_283: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_358: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_56, [1, 512, 3072]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_261: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_262: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, view_123)
    mul_263: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_262, -0.5);  mul_262 = None
    exp_20: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_263);  mul_263 = None
    mul_264: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_265: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_123, mul_264);  view_123 = mul_264 = None
    add_171: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
    mul_266: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_358, add_171);  view_358 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_359: "f32[512, 3072]" = torch.ops.aten.view.default(mul_266, [512, 3072]);  mul_266 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_359, permute_284);  permute_284 = None
    permute_285: "f32[3072, 512]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_59: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_285, view_122);  permute_285 = view_122 = None
    permute_286: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_109: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[3072]" = torch.ops.aten.view.default(sum_109, [3072]);  sum_109 = None
    permute_287: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_361: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_58, [1, 512, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_172: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_169, view_361);  add_169 = view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1], True)
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_110, [768]);  sum_110 = None
    mul_267: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, primals_41);  primals_41 = None
    mul_268: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_172, div_27);  add_172 = None
    sum_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True);  mul_268 = None
    view_363: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_124: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_27, sqrt_20);  div_27 = None
    neg_34: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_267)
    mul_269: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_34, div_124);  neg_34 = div_124 = None
    div_125: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_267, sqrt_20);  mul_267 = sqrt_20 = None
    sum_112: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    alias_65: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_270: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_65, 2);  alias_65 = None
    div_126: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_112, mul_270);  sum_112 = mul_270 = None
    neg_35: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_125)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_35, [2], True);  neg_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_70: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_126, [1, 512, 768]);  div_126 = None
    div_127: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
    pow_37: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_54, 1.0);  sub_54 = None
    mul_271: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_37, 2.0);  pow_37 = None
    mul_272: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_127, mul_271);  div_127 = mul_271 = None
    neg_36: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_272)
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_36, [2], True);  neg_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_173: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_125, mul_272);  div_125 = mul_272 = None
    add_174: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_113, sum_114);  sum_113 = sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_71: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_174, [1, 512, 768]);  add_174 = None
    div_128: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_175: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_173, div_128);  add_173 = div_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_81: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_27, full_default_1, add_175);  convert_element_type_27 = None
    mul_273: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_81, 1.1111111111111112);  where_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_364: "f32[512, 768]" = torch.ops.aten.view.default(mul_273, [512, 768]);  mul_273 = None
    mm_60: "f32[512, 768]" = torch.ops.aten.mm.default(view_364, permute_288);  permute_288 = None
    permute_289: "f32[768, 512]" = torch.ops.aten.permute.default(view_364, [1, 0])
    mm_61: "f32[768, 768]" = torch.ops.aten.mm.default(permute_289, view_120);  permute_289 = view_120 = None
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
    view_365: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_366: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_60, [1, 512, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_367: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_366, [1, 512, 12, 64]);  view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_292: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_368: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_292, [12, 512, 64]);  permute_292 = None
    bmm_44: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_293, view_368);  permute_293 = None
    bmm_45: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_368, permute_294);  view_368 = permute_294 = None
    view_369: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_44, [1, 12, 512, 64]);  bmm_44 = None
    view_370: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_45, [1, 12, 512, 512]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_82: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_26, full_default_1, view_370);  convert_element_type_26 = view_370 = None
    mul_274: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_82, 1.1111111111111112);  where_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_275: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_274, alias_68);  mul_274 = None
    sum_116: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_68, sum_116);  alias_68 = sum_116 = None
    sub_111: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_371: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_111, [12, 512, 512]);  sub_111 = None
    bmm_46: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_295, view_371);  permute_295 = None
    bmm_47: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_371, permute_296);  view_371 = permute_296 = None
    view_372: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_46, [1, 12, 64, 512]);  bmm_46 = None
    view_373: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_47, [1, 12, 512, 64]);  bmm_47 = None
    permute_297: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 1, 3, 2]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_129: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_373, full_default_2);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_117: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_369, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_298: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_117, [0, 2, 1, 3]);  sum_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_374: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_298, [1, 1, 768]);  permute_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_10: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_374, 2, 0, 9223372036854775807);  view_374 = None
    squeeze_22: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_10, 1);  slice_scatter_10 = None
    squeeze_23: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_22, 0);  squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_118: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_129, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_299: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_118, [0, 2, 1, 3]);  sum_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_375: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_299, [1, 1, 768]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_11: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_375, 2, 0, 9223372036854775807);  view_375 = None
    squeeze_24: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_11, 1);  slice_scatter_11 = None
    squeeze_25: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_24, 0);  squeeze_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_5: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_129, permute_297, view_369], 3);  div_129 = permute_297 = view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_300: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_5, [0, 2, 1, 3]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_17: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_300, memory_format = torch.contiguous_format);  permute_300 = None
    view_376: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_17, [1, 512, 2304]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_377: "f32[512, 2304]" = torch.ops.aten.view.default(view_376, [512, 2304]);  view_376 = None
    permute_301: "f32[2304, 512]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_62: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_301, view_108);  permute_301 = view_108 = None
    permute_302: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    mm_63: "f32[512, 768]" = torch.ops.aten.mm.default(view_377, permute_303);  view_377 = permute_303 = None
    view_378: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_63, [1, 512, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_176: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_175, view_378);  add_175 = view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_304: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_119: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_176, [0, 1], True)
    view_379: "f32[768]" = torch.ops.aten.view.default(sum_119, [768]);  sum_119 = None
    mul_277: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, primals_37);  primals_37 = None
    mul_278: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_176, div_24);  add_176 = None
    sum_120: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_131: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_24, sqrt_18);  div_24 = None
    neg_37: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_277)
    mul_279: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_37, div_131);  neg_37 = div_131 = None
    div_132: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_277, sqrt_18);  mul_277 = sqrt_18 = None
    sum_121: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
    alias_69: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_280: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_69, 2);  alias_69 = None
    div_133: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_121, mul_280);  sum_121 = mul_280 = None
    neg_38: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_132)
    sum_122: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_38, [2], True);  neg_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_72: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_133, [1, 512, 768]);  div_133 = None
    div_134: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_72, 768);  expand_72 = None
    pow_38: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_49, 1.0);  sub_49 = None
    mul_281: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
    mul_282: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_134, mul_281);  div_134 = mul_281 = None
    neg_39: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_282)
    sum_123: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True);  neg_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_177: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_132, mul_282);  div_132 = mul_282 = None
    add_178: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_122, sum_123);  sum_122 = sum_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_73: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_178, [1, 512, 768]);  add_178 = None
    div_135: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_73, 768);  expand_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_179: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_177, div_135);  add_177 = div_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_83: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_24, full_default_1, add_179);  convert_element_type_24 = None
    mul_283: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_83, 1.1111111111111112);  where_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_381: "f32[512, 768]" = torch.ops.aten.view.default(mul_283, [512, 768]);  mul_283 = None
    mm_64: "f32[512, 3072]" = torch.ops.aten.mm.default(view_381, permute_305);  permute_305 = None
    permute_306: "f32[768, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_65: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_306, view_106);  permute_306 = view_106 = None
    permute_307: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    permute_308: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    view_383: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_64, [1, 512, 3072]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_285: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_53, 0.5);  add_53 = None
    mul_286: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, view_105)
    mul_287: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_286, -0.5);  mul_286 = None
    exp_21: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_287);  mul_287 = None
    mul_288: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_289: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_105, mul_288);  view_105 = mul_288 = None
    add_181: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_285, mul_289);  mul_285 = mul_289 = None
    mul_290: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_383, add_181);  view_383 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_384: "f32[512, 3072]" = torch.ops.aten.view.default(mul_290, [512, 3072]);  mul_290 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_384, permute_309);  permute_309 = None
    permute_310: "f32[3072, 512]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_67: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_310, view_104);  permute_310 = view_104 = None
    permute_311: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_125: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[3072]" = torch.ops.aten.view.default(sum_125, [3072]);  sum_125 = None
    permute_312: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
    view_386: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_66, [1, 512, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_182: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_179, view_386);  add_179 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_126: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1], True)
    view_387: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    mul_291: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, primals_35);  primals_35 = None
    mul_292: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_182, div_23);  add_182 = None
    sum_127: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1], True);  mul_292 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_137: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_23, sqrt_17);  div_23 = None
    neg_40: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_291)
    mul_293: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_40, div_137);  neg_40 = div_137 = None
    div_138: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_291, sqrt_17);  mul_291 = sqrt_17 = None
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_293, [2], True);  mul_293 = None
    alias_70: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_294: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_70, 2);  alias_70 = None
    div_139: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_128, mul_294);  sum_128 = mul_294 = None
    neg_41: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_138)
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True);  neg_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_74: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_139, [1, 512, 768]);  div_139 = None
    div_140: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_74, 768);  expand_74 = None
    pow_39: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_46, 1.0);  sub_46 = None
    mul_295: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_39, 2.0);  pow_39 = None
    mul_296: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_140, mul_295);  div_140 = mul_295 = None
    neg_42: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_296)
    sum_130: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_42, [2], True);  neg_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_183: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_138, mul_296);  div_138 = mul_296 = None
    add_184: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_129, sum_130);  sum_129 = sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_75: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_184, [1, 512, 768]);  add_184 = None
    div_141: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_75, 768);  expand_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_185: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_183, div_141);  add_183 = div_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_84: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_23, full_default_1, add_185);  convert_element_type_23 = None
    mul_297: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_84, 1.1111111111111112);  where_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_389: "f32[512, 768]" = torch.ops.aten.view.default(mul_297, [512, 768]);  mul_297 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_389, permute_313);  permute_313 = None
    permute_314: "f32[768, 512]" = torch.ops.aten.permute.default(view_389, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_314, view_102);  permute_314 = view_102 = None
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
    view_391: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_68, [1, 512, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_392: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_391, [1, 512, 12, 64]);  view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_317: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_393: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_317, [12, 512, 64]);  permute_317 = None
    bmm_48: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_318, view_393);  permute_318 = None
    bmm_49: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_393, permute_319);  view_393 = permute_319 = None
    view_394: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_48, [1, 12, 512, 64]);  bmm_48 = None
    view_395: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_49, [1, 12, 512, 512]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_85: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_22, full_default_1, view_395);  convert_element_type_22 = view_395 = None
    mul_298: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_85, 1.1111111111111112);  where_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_299: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_298, alias_73);  mul_298 = None
    sum_132: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [-1], True)
    mul_300: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_73, sum_132);  alias_73 = sum_132 = None
    sub_112: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_299, mul_300);  mul_299 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_396: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_112, [12, 512, 512]);  sub_112 = None
    bmm_50: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_320, view_396);  permute_320 = None
    bmm_51: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_396, permute_321);  view_396 = permute_321 = None
    view_397: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_50, [1, 12, 64, 512]);  bmm_50 = None
    view_398: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_51, [1, 12, 512, 64]);  bmm_51 = None
    permute_322: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_397, [0, 1, 3, 2]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_142: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_398, full_default_2);  view_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_133: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_394, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_323: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_133, [0, 2, 1, 3]);  sum_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_399: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_323, [1, 1, 768]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_12: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_399, 2, 0, 9223372036854775807);  view_399 = None
    squeeze_26: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_12, 1);  slice_scatter_12 = None
    squeeze_27: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_26, 0);  squeeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_134: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_142, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_324: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_134, [0, 2, 1, 3]);  sum_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_400: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_324, [1, 1, 768]);  permute_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_13: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_400, 2, 0, 9223372036854775807);  view_400 = None
    squeeze_28: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_13, 1);  slice_scatter_13 = None
    squeeze_29: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_28, 0);  squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_6: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_142, permute_322, view_394], 3);  div_142 = permute_322 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_325: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_18: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_401: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_18, [1, 512, 2304]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_402: "f32[512, 2304]" = torch.ops.aten.view.default(view_401, [512, 2304]);  view_401 = None
    permute_326: "f32[2304, 512]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_70: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_326, view_90);  permute_326 = view_90 = None
    permute_327: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    mm_71: "f32[512, 768]" = torch.ops.aten.mm.default(view_402, permute_328);  view_402 = permute_328 = None
    view_403: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_71, [1, 512, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_186: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_185, view_403);  add_185 = view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_329: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_135: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1], True)
    view_404: "f32[768]" = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
    mul_301: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, primals_31);  primals_31 = None
    mul_302: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_186, div_20);  add_186 = None
    sum_136: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1], True);  mul_302 = None
    view_405: "f32[768]" = torch.ops.aten.view.default(sum_136, [768]);  sum_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_144: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_20, sqrt_15);  div_20 = None
    neg_43: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_301)
    mul_303: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_43, div_144);  neg_43 = div_144 = None
    div_145: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_301, sqrt_15);  mul_301 = sqrt_15 = None
    sum_137: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True);  mul_303 = None
    alias_74: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_304: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_74, 2);  alias_74 = None
    div_146: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_137, mul_304);  sum_137 = mul_304 = None
    neg_44: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_145)
    sum_138: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_44, [2], True);  neg_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_76: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_146, [1, 512, 768]);  div_146 = None
    div_147: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_76, 768);  expand_76 = None
    pow_40: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_41, 1.0);  sub_41 = None
    mul_305: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
    mul_306: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_147, mul_305);  div_147 = mul_305 = None
    neg_45: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_306)
    sum_139: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_45, [2], True);  neg_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_187: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_145, mul_306);  div_145 = mul_306 = None
    add_188: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_138, sum_139);  sum_138 = sum_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_77: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_188, [1, 512, 768]);  add_188 = None
    div_148: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_77, 768);  expand_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_189: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_187, div_148);  add_187 = div_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_86: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_20, full_default_1, add_189);  convert_element_type_20 = None
    mul_307: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_86, 1.1111111111111112);  where_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_406: "f32[512, 768]" = torch.ops.aten.view.default(mul_307, [512, 768]);  mul_307 = None
    mm_72: "f32[512, 3072]" = torch.ops.aten.mm.default(view_406, permute_330);  permute_330 = None
    permute_331: "f32[768, 512]" = torch.ops.aten.permute.default(view_406, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_331, view_88);  permute_331 = view_88 = None
    permute_332: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_140: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
    view_407: "f32[768]" = torch.ops.aten.view.default(sum_140, [768]);  sum_140 = None
    permute_333: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    view_408: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_72, [1, 512, 3072]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_309: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
    mul_310: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_311: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_310, -0.5);  mul_310 = None
    exp_22: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_311);  mul_311 = None
    mul_312: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_313: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_87, mul_312);  view_87 = mul_312 = None
    add_191: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_309, mul_313);  mul_309 = mul_313 = None
    mul_314: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_408, add_191);  view_408 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_409: "f32[512, 3072]" = torch.ops.aten.view.default(mul_314, [512, 3072]);  mul_314 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_409, permute_334);  permute_334 = None
    permute_335: "f32[3072, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_335, view_86);  permute_335 = view_86 = None
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_141: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[3072]" = torch.ops.aten.view.default(sum_141, [3072]);  sum_141 = None
    permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_411: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_74, [1, 512, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_192: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_189, view_411);  add_189 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_142: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1], True)
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    mul_315: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_192, primals_29);  primals_29 = None
    mul_316: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_192, div_19);  add_192 = None
    sum_143: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
    view_413: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_150: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_19, sqrt_14);  div_19 = None
    neg_46: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_315)
    mul_317: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_46, div_150);  neg_46 = div_150 = None
    div_151: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_315, sqrt_14);  mul_315 = sqrt_14 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
    alias_75: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_318: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_75, 2);  alias_75 = None
    div_152: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_144, mul_318);  sum_144 = mul_318 = None
    neg_47: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_151)
    sum_145: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_47, [2], True);  neg_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_78: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_152, [1, 512, 768]);  div_152 = None
    div_153: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_78, 768);  expand_78 = None
    pow_41: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_38, 1.0);  sub_38 = None
    mul_319: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_41, 2.0);  pow_41 = None
    mul_320: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_153, mul_319);  div_153 = mul_319 = None
    neg_48: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_320)
    sum_146: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_48, [2], True);  neg_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_193: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_151, mul_320);  div_151 = mul_320 = None
    add_194: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_145, sum_146);  sum_145 = sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_79: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_194, [1, 512, 768]);  add_194 = None
    div_154: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_79, 768);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_195: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_193, div_154);  add_193 = div_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_87: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_19, full_default_1, add_195);  convert_element_type_19 = None
    mul_321: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_87, 1.1111111111111112);  where_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_414: "f32[512, 768]" = torch.ops.aten.view.default(mul_321, [512, 768]);  mul_321 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_414, permute_338);  permute_338 = None
    permute_339: "f32[768, 512]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_339, view_84);  permute_339 = view_84 = None
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_341: "f32[768, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_416: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_76, [1, 512, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_417: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_416, [1, 512, 12, 64]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_342: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_418: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_342, [12, 512, 64]);  permute_342 = None
    bmm_52: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_343, view_418);  permute_343 = None
    bmm_53: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_418, permute_344);  view_418 = permute_344 = None
    view_419: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_52, [1, 12, 512, 64]);  bmm_52 = None
    view_420: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_53, [1, 12, 512, 512]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_88: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_18, full_default_1, view_420);  convert_element_type_18 = view_420 = None
    mul_322: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_88, 1.1111111111111112);  where_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_323: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_322, alias_78);  mul_322 = None
    sum_148: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [-1], True)
    mul_324: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_78, sum_148);  alias_78 = sum_148 = None
    sub_113: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_421: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_113, [12, 512, 512]);  sub_113 = None
    bmm_54: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_345, view_421);  permute_345 = None
    bmm_55: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_421, permute_346);  view_421 = permute_346 = None
    view_422: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_54, [1, 12, 64, 512]);  bmm_54 = None
    view_423: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_55, [1, 12, 512, 64]);  bmm_55 = None
    permute_347: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_422, [0, 1, 3, 2]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_155: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_423, full_default_2);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_149: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_419, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_348: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_149, [0, 2, 1, 3]);  sum_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_424: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_348, [1, 1, 768]);  permute_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_14: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_424, 2, 0, 9223372036854775807);  view_424 = None
    squeeze_30: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_14, 1);  slice_scatter_14 = None
    squeeze_31: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_30, 0);  squeeze_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_150: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_155, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_349: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_150, [0, 2, 1, 3]);  sum_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_425: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_349, [1, 1, 768]);  permute_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_15: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_425, 2, 0, 9223372036854775807);  view_425 = None
    squeeze_32: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_15, 1);  slice_scatter_15 = None
    squeeze_33: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_32, 0);  squeeze_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_7: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_155, permute_347, view_419], 3);  div_155 = permute_347 = view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_350: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_19: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
    view_426: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_19, [1, 512, 2304]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_427: "f32[512, 2304]" = torch.ops.aten.view.default(view_426, [512, 2304]);  view_426 = None
    permute_351: "f32[2304, 512]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_78: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_351, view_72);  permute_351 = view_72 = None
    permute_352: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    mm_79: "f32[512, 768]" = torch.ops.aten.mm.default(view_427, permute_353);  view_427 = permute_353 = None
    view_428: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_79, [1, 512, 768]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_196: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_195, view_428);  add_195 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_354: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_196, [0, 1], True)
    view_429: "f32[768]" = torch.ops.aten.view.default(sum_151, [768]);  sum_151 = None
    mul_325: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_196, primals_25);  primals_25 = None
    mul_326: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_196, div_16);  add_196 = None
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1], True);  mul_326 = None
    view_430: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_157: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_16, sqrt_12);  div_16 = None
    neg_49: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_325)
    mul_327: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_49, div_157);  neg_49 = div_157 = None
    div_158: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_325, sqrt_12);  mul_325 = sqrt_12 = None
    sum_153: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    alias_79: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_328: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_79, 2);  alias_79 = None
    div_159: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_153, mul_328);  sum_153 = mul_328 = None
    neg_50: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_158)
    sum_154: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_50, [2], True);  neg_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_80: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_159, [1, 512, 768]);  div_159 = None
    div_160: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_80, 768);  expand_80 = None
    pow_42: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_33, 1.0);  sub_33 = None
    mul_329: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
    mul_330: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_160, mul_329);  div_160 = mul_329 = None
    neg_51: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_330)
    sum_155: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_51, [2], True);  neg_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_197: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_158, mul_330);  div_158 = mul_330 = None
    add_198: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_154, sum_155);  sum_154 = sum_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_81: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_198, [1, 512, 768]);  add_198 = None
    div_161: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_81, 768);  expand_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_199: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_197, div_161);  add_197 = div_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_89: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_16, full_default_1, add_199);  convert_element_type_16 = None
    mul_331: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_89, 1.1111111111111112);  where_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_431: "f32[512, 768]" = torch.ops.aten.view.default(mul_331, [512, 768]);  mul_331 = None
    mm_80: "f32[512, 3072]" = torch.ops.aten.mm.default(view_431, permute_355);  permute_355 = None
    permute_356: "f32[768, 512]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_81: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_356, view_70);  permute_356 = view_70 = None
    permute_357: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_358: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
    view_433: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_80, [1, 512, 3072]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_333: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_35, 0.5);  add_35 = None
    mul_334: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, view_69)
    mul_335: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_334, -0.5);  mul_334 = None
    exp_23: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_335);  mul_335 = None
    mul_336: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_337: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_69, mul_336);  view_69 = mul_336 = None
    add_201: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_333, mul_337);  mul_333 = mul_337 = None
    mul_338: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_433, add_201);  view_433 = add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_434: "f32[512, 3072]" = torch.ops.aten.view.default(mul_338, [512, 3072]);  mul_338 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_434, permute_359);  permute_359 = None
    permute_360: "f32[3072, 512]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_83: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_360, view_68);  permute_360 = view_68 = None
    permute_361: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_157: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[3072]" = torch.ops.aten.view.default(sum_157, [3072]);  sum_157 = None
    permute_362: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    view_436: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_82, [1, 512, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_202: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_199, view_436);  add_199 = view_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_158: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1], True)
    view_437: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    mul_339: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_202, primals_23);  primals_23 = None
    mul_340: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_202, div_15);  add_202 = None
    sum_159: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1], True);  mul_340 = None
    view_438: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_163: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_15, sqrt_11);  div_15 = None
    neg_52: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_339)
    mul_341: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_52, div_163);  neg_52 = div_163 = None
    div_164: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_339, sqrt_11);  mul_339 = sqrt_11 = None
    sum_160: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    alias_80: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_342: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_80, 2);  alias_80 = None
    div_165: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_160, mul_342);  sum_160 = mul_342 = None
    neg_53: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_164)
    sum_161: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_53, [2], True);  neg_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_82: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_165, [1, 512, 768]);  div_165 = None
    div_166: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_82, 768);  expand_82 = None
    pow_43: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_30, 1.0);  sub_30 = None
    mul_343: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_43, 2.0);  pow_43 = None
    mul_344: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_166, mul_343);  div_166 = mul_343 = None
    neg_54: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_344)
    sum_162: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_54, [2], True);  neg_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_203: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_164, mul_344);  div_164 = mul_344 = None
    add_204: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_161, sum_162);  sum_161 = sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_83: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_204, [1, 512, 768]);  add_204 = None
    div_167: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_83, 768);  expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_205: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_203, div_167);  add_203 = div_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_90: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_15, full_default_1, add_205);  convert_element_type_15 = None
    mul_345: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_90, 1.1111111111111112);  where_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_439: "f32[512, 768]" = torch.ops.aten.view.default(mul_345, [512, 768]);  mul_345 = None
    mm_84: "f32[512, 768]" = torch.ops.aten.mm.default(view_439, permute_363);  permute_363 = None
    permute_364: "f32[768, 512]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_85: "f32[768, 768]" = torch.ops.aten.mm.default(permute_364, view_66);  permute_364 = view_66 = None
    permute_365: "f32[768, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_366: "f32[768, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    view_441: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_84, [1, 512, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_442: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_441, [1, 512, 12, 64]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_367: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_443: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_367, [12, 512, 64]);  permute_367 = None
    bmm_56: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_368, view_443);  permute_368 = None
    bmm_57: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_443, permute_369);  view_443 = permute_369 = None
    view_444: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_56, [1, 12, 512, 64]);  bmm_56 = None
    view_445: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_57, [1, 12, 512, 512]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_91: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_14, full_default_1, view_445);  convert_element_type_14 = view_445 = None
    mul_346: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_91, 1.1111111111111112);  where_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_347: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_346, alias_83);  mul_346 = None
    sum_164: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [-1], True)
    mul_348: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_83, sum_164);  alias_83 = sum_164 = None
    sub_114: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_446: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_114, [12, 512, 512]);  sub_114 = None
    bmm_58: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_370, view_446);  permute_370 = None
    bmm_59: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_446, permute_371);  view_446 = permute_371 = None
    view_447: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_58, [1, 12, 64, 512]);  bmm_58 = None
    view_448: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_59, [1, 12, 512, 64]);  bmm_59 = None
    permute_372: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_168: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_448, full_default_2);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_165: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_444, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_373: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_165, [0, 2, 1, 3]);  sum_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_449: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_373, [1, 1, 768]);  permute_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_16: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_449, 2, 0, 9223372036854775807);  view_449 = None
    squeeze_34: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_16, 1);  slice_scatter_16 = None
    squeeze_35: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_34, 0);  squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_166: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_168, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_374: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_166, [0, 2, 1, 3]);  sum_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_450: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_374, [1, 1, 768]);  permute_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_17: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_450, 2, 0, 9223372036854775807);  view_450 = None
    squeeze_36: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_17, 1);  slice_scatter_17 = None
    squeeze_37: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_36, 0);  squeeze_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_8: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_168, permute_372, view_444], 3);  div_168 = permute_372 = view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_375: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_8, [0, 2, 1, 3]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    view_451: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_20, [1, 512, 2304]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_452: "f32[512, 2304]" = torch.ops.aten.view.default(view_451, [512, 2304]);  view_451 = None
    permute_376: "f32[2304, 512]" = torch.ops.aten.permute.default(view_452, [1, 0])
    mm_86: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_376, view_54);  permute_376 = view_54 = None
    permute_377: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    mm_87: "f32[512, 768]" = torch.ops.aten.mm.default(view_452, permute_378);  view_452 = permute_378 = None
    view_453: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_87, [1, 512, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_206: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_205, view_453);  add_205 = view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_379: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1], True)
    view_454: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    mul_349: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_206, primals_19);  primals_19 = None
    mul_350: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_206, div_12);  add_206 = None
    sum_168: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1], True);  mul_350 = None
    view_455: "f32[768]" = torch.ops.aten.view.default(sum_168, [768]);  sum_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_170: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_12, sqrt_9);  div_12 = None
    neg_55: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_349)
    mul_351: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_55, div_170);  neg_55 = div_170 = None
    div_171: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_349, sqrt_9);  mul_349 = sqrt_9 = None
    sum_169: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
    alias_84: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_352: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_84, 2);  alias_84 = None
    div_172: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_169, mul_352);  sum_169 = mul_352 = None
    neg_56: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_171)
    sum_170: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_56, [2], True);  neg_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_84: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_172, [1, 512, 768]);  div_172 = None
    div_173: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_84, 768);  expand_84 = None
    pow_44: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_25, 1.0);  sub_25 = None
    mul_353: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
    mul_354: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_173, mul_353);  div_173 = mul_353 = None
    neg_57: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_354)
    sum_171: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_57, [2], True);  neg_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_207: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_171, mul_354);  div_171 = mul_354 = None
    add_208: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_170, sum_171);  sum_170 = sum_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_85: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_208, [1, 512, 768]);  add_208 = None
    div_174: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_85, 768);  expand_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_209: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_207, div_174);  add_207 = div_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_92: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_12, full_default_1, add_209);  convert_element_type_12 = None
    mul_355: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_92, 1.1111111111111112);  where_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_456: "f32[512, 768]" = torch.ops.aten.view.default(mul_355, [512, 768]);  mul_355 = None
    mm_88: "f32[512, 3072]" = torch.ops.aten.mm.default(view_456, permute_380);  permute_380 = None
    permute_381: "f32[768, 512]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_89: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_381, view_52);  permute_381 = view_52 = None
    permute_382: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_383: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_458: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_88, [1, 512, 3072]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_357: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
    mul_358: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, view_51)
    mul_359: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_358, -0.5);  mul_358 = None
    exp_24: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_359);  mul_359 = None
    mul_360: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_361: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_51, mul_360);  view_51 = mul_360 = None
    add_211: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_357, mul_361);  mul_357 = mul_361 = None
    mul_362: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_458, add_211);  view_458 = add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_459: "f32[512, 3072]" = torch.ops.aten.view.default(mul_362, [512, 3072]);  mul_362 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_384);  permute_384 = None
    permute_385: "f32[3072, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_91: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_385, view_50);  permute_385 = view_50 = None
    permute_386: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_173: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[3072]" = torch.ops.aten.view.default(sum_173, [3072]);  sum_173 = None
    permute_387: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_386, [1, 0]);  permute_386 = None
    view_461: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_90, [1, 512, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_212: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_209, view_461);  add_209 = view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1], True)
    view_462: "f32[768]" = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
    mul_363: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_212, primals_17);  primals_17 = None
    mul_364: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_212, div_11);  add_212 = None
    sum_175: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 1], True);  mul_364 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_175, [768]);  sum_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_176: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_11, sqrt_8);  div_11 = None
    neg_58: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_363)
    mul_365: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_58, div_176);  neg_58 = div_176 = None
    div_177: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_363, sqrt_8);  mul_363 = sqrt_8 = None
    sum_176: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True);  mul_365 = None
    alias_85: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_366: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_85, 2);  alias_85 = None
    div_178: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_176, mul_366);  sum_176 = mul_366 = None
    neg_59: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_177)
    sum_177: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_59, [2], True);  neg_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_86: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_178, [1, 512, 768]);  div_178 = None
    div_179: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_86, 768);  expand_86 = None
    pow_45: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_22, 1.0);  sub_22 = None
    mul_367: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_45, 2.0);  pow_45 = None
    mul_368: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_179, mul_367);  div_179 = mul_367 = None
    neg_60: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_368)
    sum_178: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_60, [2], True);  neg_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_213: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_177, mul_368);  div_177 = mul_368 = None
    add_214: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_177, sum_178);  sum_177 = sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_87: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_214, [1, 512, 768]);  add_214 = None
    div_180: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_87, 768);  expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_215: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_213, div_180);  add_213 = div_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_93: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_11, full_default_1, add_215);  convert_element_type_11 = None
    mul_369: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_93, 1.1111111111111112);  where_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_464: "f32[512, 768]" = torch.ops.aten.view.default(mul_369, [512, 768]);  mul_369 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_464, permute_388);  permute_388 = None
    permute_389: "f32[768, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_389, view_48);  permute_389 = view_48 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_391: "f32[768, 768]" = torch.ops.aten.permute.default(permute_390, [1, 0]);  permute_390 = None
    view_466: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_92, [1, 512, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_467: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_466, [1, 512, 12, 64]);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_392: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_468: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_392, [12, 512, 64]);  permute_392 = None
    bmm_60: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_393, view_468);  permute_393 = None
    bmm_61: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_468, permute_394);  view_468 = permute_394 = None
    view_469: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_60, [1, 12, 512, 64]);  bmm_60 = None
    view_470: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_61, [1, 12, 512, 512]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_94: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_10, full_default_1, view_470);  convert_element_type_10 = view_470 = None
    mul_370: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_94, 1.1111111111111112);  where_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_371: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_370, alias_88);  mul_370 = None
    sum_180: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [-1], True)
    mul_372: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_88, sum_180);  alias_88 = sum_180 = None
    sub_115: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_471: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_115, [12, 512, 512]);  sub_115 = None
    bmm_62: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_395, view_471);  permute_395 = None
    bmm_63: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_471, permute_396);  view_471 = permute_396 = None
    view_472: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_62, [1, 12, 64, 512]);  bmm_62 = None
    view_473: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_63, [1, 12, 512, 64]);  bmm_63 = None
    permute_397: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_472, [0, 1, 3, 2]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_181: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_473, full_default_2);  view_473 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_181: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_469, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_398: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_181, [0, 2, 1, 3]);  sum_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_474: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_398, [1, 1, 768]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_18: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_474, 2, 0, 9223372036854775807);  view_474 = None
    squeeze_38: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_18, 1);  slice_scatter_18 = None
    squeeze_39: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_38, 0);  squeeze_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_182: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_181, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_399: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_182, [0, 2, 1, 3]);  sum_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_475: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_399, [1, 1, 768]);  permute_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_19: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_475, 2, 0, 9223372036854775807);  view_475 = None
    squeeze_40: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_19, 1);  slice_scatter_19 = None
    squeeze_41: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_40, 0);  squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_9: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_181, permute_397, view_469], 3);  div_181 = permute_397 = view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_400: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_9, [0, 2, 1, 3]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_400, memory_format = torch.contiguous_format);  permute_400 = None
    view_476: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_21, [1, 512, 2304]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_477: "f32[512, 2304]" = torch.ops.aten.view.default(view_476, [512, 2304]);  view_476 = None
    permute_401: "f32[2304, 512]" = torch.ops.aten.permute.default(view_477, [1, 0])
    mm_94: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_401, view_36);  permute_401 = view_36 = None
    permute_402: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_94, [1, 0]);  mm_94 = None
    mm_95: "f32[512, 768]" = torch.ops.aten.mm.default(view_477, permute_403);  view_477 = permute_403 = None
    view_478: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_95, [1, 512, 768]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_216: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_215, view_478);  add_215 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_404: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_183: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_216, [0, 1], True)
    view_479: "f32[768]" = torch.ops.aten.view.default(sum_183, [768]);  sum_183 = None
    mul_373: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, primals_13);  primals_13 = None
    mul_374: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_216, div_8);  add_216 = None
    sum_184: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1], True);  mul_374 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_183: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_8, sqrt_6);  div_8 = None
    neg_61: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_373)
    mul_375: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_61, div_183);  neg_61 = div_183 = None
    div_184: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_373, sqrt_6);  mul_373 = sqrt_6 = None
    sum_185: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_375, [2], True);  mul_375 = None
    alias_89: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_376: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_89, 2);  alias_89 = None
    div_185: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_185, mul_376);  sum_185 = mul_376 = None
    neg_62: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_184)
    sum_186: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_62, [2], True);  neg_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_88: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_185, [1, 512, 768]);  div_185 = None
    div_186: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_88, 768);  expand_88 = None
    pow_46: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_17, 1.0);  sub_17 = None
    mul_377: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
    mul_378: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_186, mul_377);  div_186 = mul_377 = None
    neg_63: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_378)
    sum_187: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_63, [2], True);  neg_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_217: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_184, mul_378);  div_184 = mul_378 = None
    add_218: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_186, sum_187);  sum_186 = sum_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_89: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_218, [1, 512, 768]);  add_218 = None
    div_187: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_89, 768);  expand_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_219: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_217, div_187);  add_217 = div_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_95: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_8, full_default_1, add_219);  convert_element_type_8 = None
    mul_379: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_95, 1.1111111111111112);  where_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 768]" = torch.ops.aten.view.default(mul_379, [512, 768]);  mul_379 = None
    mm_96: "f32[512, 3072]" = torch.ops.aten.mm.default(view_481, permute_405);  permute_405 = None
    permute_406: "f32[768, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_406, view_34);  permute_406 = view_34 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    permute_408: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
    view_483: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_96, [1, 512, 3072]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_381: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_17, 0.5);  add_17 = None
    mul_382: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_383: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
    exp_25: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_383);  mul_383 = None
    mul_384: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_385: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_33, mul_384);  view_33 = mul_384 = None
    add_221: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
    mul_386: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_483, add_221);  view_483 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_484: "f32[512, 3072]" = torch.ops.aten.view.default(mul_386, [512, 3072]);  mul_386 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_484, permute_409);  permute_409 = None
    permute_410: "f32[3072, 512]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_410, view_32);  permute_410 = view_32 = None
    permute_411: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_189: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[3072]" = torch.ops.aten.view.default(sum_189, [3072]);  sum_189 = None
    permute_412: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
    view_486: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_98, [1, 512, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_222: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_219, view_486);  add_219 = view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_190: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1], True)
    view_487: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    mul_387: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_11);  primals_11 = None
    mul_388: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_222, div_7);  add_222 = None
    sum_191: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True);  mul_388 = None
    view_488: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_189: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_7, sqrt_5);  div_7 = None
    neg_64: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_387)
    mul_389: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_64, div_189);  neg_64 = div_189 = None
    div_190: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_387, sqrt_5);  mul_387 = sqrt_5 = None
    sum_192: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    alias_90: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_390: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_90, 2);  alias_90 = None
    div_191: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_192, mul_390);  sum_192 = mul_390 = None
    neg_65: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_190)
    sum_193: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_65, [2], True);  neg_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_90: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_191, [1, 512, 768]);  div_191 = None
    div_192: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_90, 768);  expand_90 = None
    pow_47: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_14, 1.0);  sub_14 = None
    mul_391: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_47, 2.0);  pow_47 = None
    mul_392: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_192, mul_391);  div_192 = mul_391 = None
    neg_66: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_392)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_66, [2], True);  neg_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_223: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_190, mul_392);  div_190 = mul_392 = None
    add_224: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_193, sum_194);  sum_193 = sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_91: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_224, [1, 512, 768]);  add_224 = None
    div_193: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_91, 768);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_225: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_223, div_193);  add_223 = div_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_96: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_7, full_default_1, add_225);  convert_element_type_7 = None
    mul_393: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_96, 1.1111111111111112);  where_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_489: "f32[512, 768]" = torch.ops.aten.view.default(mul_393, [512, 768]);  mul_393 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_489, permute_413);  permute_413 = None
    permute_414: "f32[768, 512]" = torch.ops.aten.permute.default(view_489, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_414, view_30);  permute_414 = view_30 = None
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
    view_490: "f32[768]" = torch.ops.aten.view.default(sum_195, [768]);  sum_195 = None
    permute_416: "f32[768, 768]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
    view_491: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_100, [1, 512, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_492: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_491, [1, 512, 12, 64]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_417: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_493: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_417, [12, 512, 64]);  permute_417 = None
    bmm_64: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_418, view_493);  permute_418 = None
    bmm_65: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_493, permute_419);  view_493 = permute_419 = None
    view_494: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_64, [1, 12, 512, 64]);  bmm_64 = None
    view_495: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_65, [1, 12, 512, 512]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_97: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_6, full_default_1, view_495);  convert_element_type_6 = view_495 = None
    mul_394: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_97, 1.1111111111111112);  where_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_395: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_394, alias_93);  mul_394 = None
    sum_196: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [-1], True)
    mul_396: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_93, sum_196);  alias_93 = sum_196 = None
    sub_116: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_496: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_116, [12, 512, 512]);  sub_116 = None
    bmm_66: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_420, view_496);  permute_420 = None
    bmm_67: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_496, permute_421);  view_496 = permute_421 = None
    view_497: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_66, [1, 12, 64, 512]);  bmm_66 = None
    view_498: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_67, [1, 12, 512, 64]);  bmm_67 = None
    permute_422: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_497, [0, 1, 3, 2]);  view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_194: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_498, full_default_2);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_197: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_494, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_423: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_197, [0, 2, 1, 3]);  sum_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_499: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_423, [1, 1, 768]);  permute_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_20: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_499, 2, 0, 9223372036854775807);  view_499 = None
    squeeze_42: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_20, 1);  slice_scatter_20 = None
    squeeze_43: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_42, 0);  squeeze_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_198: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_194, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_424: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_198, [0, 2, 1, 3]);  sum_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_500: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_424, [1, 1, 768]);  permute_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_21: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_500, 2, 0, 9223372036854775807);  view_500 = None
    squeeze_44: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_21, 1);  slice_scatter_21 = None
    squeeze_45: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_44, 0);  squeeze_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_10: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_194, permute_422, view_494], 3);  div_194 = permute_422 = view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_22: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_501: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_22, [1, 512, 2304]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_502: "f32[512, 2304]" = torch.ops.aten.view.default(view_501, [512, 2304]);  view_501 = None
    permute_426: "f32[2304, 512]" = torch.ops.aten.permute.default(view_502, [1, 0])
    mm_102: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_426, view_18);  permute_426 = view_18 = None
    permute_427: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    mm_103: "f32[512, 768]" = torch.ops.aten.mm.default(view_502, permute_428);  view_502 = permute_428 = None
    view_503: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_103, [1, 512, 768]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_226: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_225, view_503);  add_225 = view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_429: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_199: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1], True)
    view_504: "f32[768]" = torch.ops.aten.view.default(sum_199, [768]);  sum_199 = None
    mul_397: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_226, primals_7);  primals_7 = None
    mul_398: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_226, div_4);  add_226 = None
    sum_200: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1], True);  mul_398 = None
    view_505: "f32[768]" = torch.ops.aten.view.default(sum_200, [768]);  sum_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_196: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_4, sqrt_3);  div_4 = None
    neg_67: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_397)
    mul_399: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_67, div_196);  neg_67 = div_196 = None
    div_197: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_397, sqrt_3);  mul_397 = sqrt_3 = None
    sum_201: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True);  mul_399 = None
    alias_94: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_400: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_94, 2);  alias_94 = None
    div_198: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_201, mul_400);  sum_201 = mul_400 = None
    neg_68: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_197)
    sum_202: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_68, [2], True);  neg_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_92: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_198, [1, 512, 768]);  div_198 = None
    div_199: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_92, 768);  expand_92 = None
    pow_48: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_9, 1.0);  sub_9 = None
    mul_401: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
    mul_402: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_199, mul_401);  div_199 = mul_401 = None
    neg_69: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_402)
    sum_203: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_69, [2], True);  neg_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_227: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_197, mul_402);  div_197 = mul_402 = None
    add_228: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_202, sum_203);  sum_202 = sum_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_93: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_228, [1, 512, 768]);  add_228 = None
    div_200: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_93, 768);  expand_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_229: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_227, div_200);  add_227 = div_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_98: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_4, full_default_1, add_229);  convert_element_type_4 = None
    mul_403: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_98, 1.1111111111111112);  where_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:361, code: hidden_states = self.dense(hidden_states)
    view_506: "f32[512, 768]" = torch.ops.aten.view.default(mul_403, [512, 768]);  mul_403 = None
    mm_104: "f32[512, 3072]" = torch.ops.aten.mm.default(view_506, permute_430);  permute_430 = None
    permute_431: "f32[768, 512]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_105: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_431, view_16);  permute_431 = view_16 = None
    permute_432: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_204: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[768]" = torch.ops.aten.view.default(sum_204, [768]);  sum_204 = None
    permute_433: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_508: "f32[1, 512, 3072]" = torch.ops.aten.view.default(mm_104, [1, 512, 3072]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_405: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_406: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, view_15)
    mul_407: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_26: "f32[1, 512, 3072]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_409: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_15, mul_408);  view_15 = mul_408 = None
    add_231: "f32[1, 512, 3072]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[1, 512, 3072]" = torch.ops.aten.mul.Tensor(view_508, add_231);  view_508 = add_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    view_509: "f32[512, 3072]" = torch.ops.aten.view.default(mul_410, [512, 3072]);  mul_410 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_509, permute_434);  permute_434 = None
    permute_435: "f32[3072, 512]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_107: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_435, view_14);  permute_435 = view_14 = None
    permute_436: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_205: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[3072]" = torch.ops.aten.view.default(sum_205, [3072]);  sum_205 = None
    permute_437: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
    view_511: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_106, [1, 512, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:347, code: hidden_states = self.dense(hidden_states)
    add_232: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_229, view_511);  add_229 = view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_206: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_232, [0, 1], True)
    view_512: "f32[768]" = torch.ops.aten.view.default(sum_206, [768]);  sum_206 = None
    mul_411: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_232, primals_5);  primals_5 = None
    mul_412: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(add_232, div_3);  add_232 = None
    sum_207: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 1], True);  mul_412 = None
    view_513: "f32[768]" = torch.ops.aten.view.default(sum_207, [768]);  sum_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_202: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div_3, sqrt_2);  div_3 = None
    neg_70: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_411)
    mul_413: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_70, div_202);  neg_70 = div_202 = None
    div_203: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_411, sqrt_2);  mul_411 = sqrt_2 = None
    sum_208: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
    alias_95: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_414: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_95, 2);  alias_95 = None
    div_204: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_208, mul_414);  sum_208 = mul_414 = None
    neg_71: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_203)
    sum_209: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_71, [2], True);  neg_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_94: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_204, [1, 512, 768]);  div_204 = None
    div_205: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_94, 768);  expand_94 = None
    pow_49: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub_6, 1.0);  sub_6 = None
    mul_415: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_49, 2.0);  pow_49 = None
    mul_416: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_205, mul_415);  div_205 = mul_415 = None
    neg_72: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_416)
    sum_210: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_72, [2], True);  neg_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_233: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_203, mul_416);  div_203 = mul_416 = None
    add_234: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_209, sum_210);  sum_209 = sum_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_95: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_234, [1, 512, 768]);  add_234 = None
    div_206: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_95, 768);  expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_235: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_233, div_206);  add_233 = div_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_99: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type_3, full_default_1, add_235);  convert_element_type_3 = None
    mul_417: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_99, 1.1111111111111112);  where_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:294, code: hidden_states = self.dense(hidden_states)
    view_514: "f32[512, 768]" = torch.ops.aten.view.default(mul_417, [512, 768]);  mul_417 = None
    mm_108: "f32[512, 768]" = torch.ops.aten.mm.default(view_514, permute_438);  permute_438 = None
    permute_439: "f32[768, 512]" = torch.ops.aten.permute.default(view_514, [1, 0])
    mm_109: "f32[768, 768]" = torch.ops.aten.mm.default(permute_439, view_12);  permute_439 = view_12 = None
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_211: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_514, [0], True);  view_514 = None
    view_515: "f32[768]" = torch.ops.aten.view.default(sum_211, [768]);  sum_211 = None
    permute_441: "f32[768, 768]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
    view_516: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_108, [1, 512, 768]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:684, code: context_layer = context_layer.view(new_context_layer_shape)
    view_517: "f32[1, 512, 12, 64]" = torch.ops.aten.view.default(view_516, [1, 512, 12, 64]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:682, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_442: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:681, code: context_layer = torch.matmul(attention_probs, value_layer)
    view_518: "f32[12, 512, 64]" = torch.ops.aten.view.default(permute_442, [12, 512, 64]);  permute_442 = None
    bmm_68: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(permute_443, view_518);  permute_443 = None
    bmm_69: "f32[12, 512, 512]" = torch.ops.aten.bmm.default(view_518, permute_444);  view_518 = permute_444 = None
    view_519: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_68, [1, 12, 512, 64]);  bmm_68 = None
    view_520: "f32[1, 12, 512, 512]" = torch.ops.aten.view.default(bmm_69, [1, 12, 512, 512]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_100: "f32[1, 12, 512, 512]" = torch.ops.aten.where.self(convert_element_type_2, full_default_1, view_520);  convert_element_type_2 = view_520 = None
    mul_418: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(where_100, 1.1111111111111112);  where_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:676, code: attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
    mul_419: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(mul_418, alias_98);  mul_418 = None
    sum_212: "f32[1, 12, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
    mul_420: "f32[1, 12, 512, 512]" = torch.ops.aten.mul.Tensor(alias_98, sum_212);  alias_98 = sum_212 = None
    sub_117: "f32[1, 12, 512, 512]" = torch.ops.aten.sub.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:664, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    view_521: "f32[12, 512, 512]" = torch.ops.aten.view.default(sub_117, [12, 512, 512]);  sub_117 = None
    bmm_70: "f32[12, 64, 512]" = torch.ops.aten.bmm.default(permute_445, view_521);  permute_445 = None
    bmm_71: "f32[12, 512, 64]" = torch.ops.aten.bmm.default(view_521, permute_446);  view_521 = permute_446 = None
    view_522: "f32[1, 12, 64, 512]" = torch.ops.aten.view.default(bmm_70, [1, 12, 64, 512]);  bmm_70 = None
    view_523: "f32[1, 12, 512, 64]" = torch.ops.aten.view.default(bmm_71, [1, 12, 512, 64]);  bmm_71 = None
    permute_447: "f32[1, 12, 512, 64]" = torch.ops.aten.permute.default(view_522, [0, 1, 3, 2]);  view_522 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:663, code: query_layer = query_layer / scale.to(dtype=query_layer.dtype)
    div_207: "f32[1, 12, 512, 64]" = torch.ops.aten.div.Tensor(view_523, full_default_2);  view_523 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    sum_213: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(view_519, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_448: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_213, [0, 2, 1, 3]);  sum_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_524: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_448, [1, 1, 768]);  permute_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:657, code: value_layer = value_layer + self.transpose_for_scores(self.v_bias[None, None, :])
    slice_scatter_22: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_524, 2, 0, 9223372036854775807);  view_524 = None
    squeeze_46: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_22, 1);  slice_scatter_22 = None
    squeeze_47: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_46, 0);  squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    sum_214: "f32[1, 12, 1, 64]" = torch.ops.aten.sum.dim_IntList(div_207, [2], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_449: "f32[1, 1, 12, 64]" = torch.ops.aten.permute.default(sum_214, [0, 2, 1, 3]);  sum_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    view_525: "f32[1, 1, 768]" = torch.ops.aten.view.default(permute_449, [1, 1, 768]);  permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:656, code: query_layer = query_layer + self.transpose_for_scores(self.q_bias[None, None, :])
    slice_scatter_23: "f32[1, 1, 768]" = torch.ops.aten.slice_scatter.default(full_default_94, view_525, 2, 0, 9223372036854775807);  full_default_94 = view_525 = None
    squeeze_48: "f32[1, 768]" = torch.ops.aten.squeeze.dim(slice_scatter_23, 1);  slice_scatter_23 = None
    squeeze_49: "f32[768]" = torch.ops.aten.squeeze.dim(squeeze_48, 0);  squeeze_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:639, code: query_layer, key_layer, value_layer = self.transpose_for_scores(qp).chunk(3, dim=-1)
    cat_11: "f32[1, 12, 512, 192]" = torch.ops.aten.cat.default([div_207, permute_447, view_519], 3);  div_207 = permute_447 = view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:597, code: return x.permute(0, 2, 1, 3)
    permute_450: "f32[1, 512, 12, 192]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:596, code: x = x.view(new_x_shape)
    clone_23: "f32[1, 512, 12, 192]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
    view_526: "f32[1, 512, 2304]" = torch.ops.aten.view.default(clone_23, [1, 512, 2304]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    view_527: "f32[512, 2304]" = torch.ops.aten.view.default(view_526, [512, 2304]);  view_526 = None
    permute_451: "f32[2304, 512]" = torch.ops.aten.permute.default(view_527, [1, 0])
    mm_110: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_451, view);  permute_451 = view = None
    permute_452: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    mm_111: "f32[512, 768]" = torch.ops.aten.mm.default(view_527, permute_453);  view_527 = permute_453 = None
    view_528: "f32[1, 512, 768]" = torch.ops.aten.view.default(mm_111, [1, 512, 768]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    add_236: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_235, view_528);  add_235 = view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:638, code: qp = self.in_proj(hidden_states)  # .split(self.all_head_size, dim=-1)
    permute_454: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:239, code: return XDropout.apply(x, self.get_context())
    where_101: "f32[1, 512, 768]" = torch.ops.aten.where.self(convert_element_type, full_default_1, add_236);  convert_element_type = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:812, code: embeddings = embeddings * mask
    mul_421: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(where_101, 1.1111111111111112);  where_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:282, code: y = self.weight * hidden_states + self.bias
    sum_215: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1], True)
    view_529: "f32[768]" = torch.ops.aten.view.default(sum_215, [768]);  sum_215 = None
    mul_423: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_421, primals_1);  primals_1 = None
    mul_424: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(mul_421, div);  mul_421 = None
    sum_216: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_424, [0, 1], True);  mul_424 = None
    view_530: "f32[768]" = torch.ops.aten.view.default(sum_216, [768]);  sum_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:280, code: hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
    div_209: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(div, sqrt);  div = None
    neg_73: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_423)
    mul_425: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(neg_73, div_209);  neg_73 = div_209 = None
    div_210: "f32[1, 512, 768]" = torch.ops.aten.div.Tensor(mul_423, sqrt);  mul_423 = sqrt = None
    sum_217: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
    alias_99: "f32[1, 512, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_426: "f32[1, 512, 1]" = torch.ops.aten.mul.Scalar(alias_99, 2);  alias_99 = None
    div_211: "f32[1, 512, 1]" = torch.ops.aten.div.Tensor(sum_217, mul_426);  sum_217 = mul_426 = None
    neg_74: "f32[1, 512, 768]" = torch.ops.aten.neg.default(div_210)
    sum_218: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_74, [2], True);  neg_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    expand_96: "f32[1, 512, 768]" = torch.ops.aten.expand.default(div_211, [1, 512, 768]);  div_211 = None
    div_212: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_96, 768);  expand_96 = None
    pow_50: "f32[1, 512, 768]" = torch.ops.aten.pow.Tensor_Scalar(sub, 1.0);  sub = None
    mul_427: "f32[1, 512, 768]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
    mul_428: "f32[1, 512, 768]" = torch.ops.aten.mul.Tensor(div_212, mul_427);  div_212 = mul_427 = None
    neg_75: "f32[1, 512, 768]" = torch.ops.aten.neg.default(mul_428)
    sum_219: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(neg_75, [2], True);  neg_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:279, code: variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    add_237: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(div_210, mul_428);  div_210 = mul_428 = None
    add_238: "f32[1, 512, 1]" = torch.ops.aten.add.Tensor(sum_218, sum_219);  sum_218 = sum_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    expand_97: "f32[1, 512, 768]" = torch.ops.aten.expand.default(add_238, [1, 512, 768]);  add_238 = None
    div_213: "f32[1, 512, 768]" = torch.ops.aten.div.Scalar(expand_97, 768);  expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:278, code: mean = hidden_states.mean(-1, keepdim=True)
    add_239: "f32[1, 512, 768]" = torch.ops.aten.add.Tensor(add_237, div_213);  add_237 = div_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:789, code: position_embeddings = self.position_embeddings(position_ids.long())
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_1, -1)
    unsqueeze_54: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_102: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_54, full_default_1, add_239);  unsqueeze_54 = None
    full_default_153: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_153, [slice_1], where_102, True);  full_default_153 = slice_1 = where_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/deberta/modeling_deberta.py:786, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_168, 0)
    unsqueeze_55: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_103: "f32[1, 512, 768]" = torch.ops.aten.where.self(unsqueeze_55, full_default_1, add_239);  unsqueeze_55 = full_default_1 = add_239 = None
    full_default_155: "f32[50265, 768]" = torch.ops.aten.full.default([50265, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[50265, 768]" = torch.ops.aten._unsafe_index_put.default(full_default_155, [primals_168], where_103, True);  full_default_155 = primals_168 = where_103 = None
    return [view_530, view_529, squeeze_49, squeeze_47, view_513, view_512, view_505, view_504, squeeze_45, squeeze_43, view_488, view_487, view_480, view_479, squeeze_41, squeeze_39, view_463, view_462, view_455, view_454, squeeze_37, squeeze_35, view_438, view_437, view_430, view_429, squeeze_33, squeeze_31, view_413, view_412, view_405, view_404, squeeze_29, squeeze_27, view_388, view_387, view_380, view_379, squeeze_25, squeeze_23, view_363, view_362, view_355, view_354, squeeze_21, squeeze_19, view_338, view_337, view_330, view_329, squeeze_17, squeeze_15, view_313, view_312, view_305, view_304, squeeze_13, squeeze_11, view_288, view_287, view_280, view_279, squeeze_9, squeeze_7, view_263, view_262, view_255, view_254, squeeze_5, squeeze_3, view_238, view_237, view_230, view_229, _unsafe_index_put_1, _unsafe_index_put, permute_454, permute_441, view_515, permute_437, view_510, permute_433, view_507, permute_429, permute_416, view_490, permute_412, view_485, permute_408, view_482, permute_404, permute_391, view_465, permute_387, view_460, permute_383, view_457, permute_379, permute_366, view_440, permute_362, view_435, permute_358, view_432, permute_354, permute_341, view_415, permute_337, view_410, permute_333, view_407, permute_329, permute_316, view_390, permute_312, view_385, permute_308, view_382, permute_304, permute_291, view_365, permute_287, view_360, permute_283, view_357, permute_279, permute_266, view_340, permute_262, view_335, permute_258, view_332, permute_254, permute_241, view_315, permute_237, view_310, permute_233, view_307, permute_229, permute_216, view_290, permute_212, view_285, permute_208, view_282, permute_204, permute_191, view_265, permute_187, view_260, permute_183, view_257, permute_179, permute_166, view_240, permute_162, view_235, permute_158, view_232, permute_154, view_227, sum_20, sum_21, permute_150, view_224, None, None, None]
    