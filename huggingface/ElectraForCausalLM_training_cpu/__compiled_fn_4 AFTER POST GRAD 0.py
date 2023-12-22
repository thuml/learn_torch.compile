from __future__ import annotations



def forward(self, primals_4: "f32[128]", primals_16: "f32[256]", primals_22: "f32[256]", primals_32: "f32[256]", primals_38: "f32[256]", primals_48: "f32[256]", primals_54: "f32[256]", primals_64: "f32[256]", primals_70: "f32[256]", primals_80: "f32[256]", primals_86: "f32[256]", primals_96: "f32[256]", primals_102: "f32[256]", primals_112: "f32[256]", primals_118: "f32[256]", primals_128: "f32[256]", primals_134: "f32[256]", primals_144: "f32[256]", primals_150: "f32[256]", primals_160: "f32[256]", primals_166: "f32[256]", primals_176: "f32[256]", primals_182: "f32[256]", primals_192: "f32[256]", primals_198: "f32[256]", primals_202: "f32[128]", primals_209: "i64[1, 512]", expand: "i64[1, 512]", slice_4: "i64[1, 512]", mul_1: "f32[1, 512, 128]", getitem_3: "b8[1, 512, 128]", view: "f32[512, 128]", view_2: "f32[512, 256]", getitem_149: "b8[1, 4, 512, 512]", permute_default_67: "f32[4, 512, 512]", permute_default_68: "f32[4, 64, 512]", alias_default_23: "f32[1, 4, 512, 512]", permute_default_69: "f32[4, 64, 512]", permute_default_70: "f32[4, 512, 64]", view_18: "f32[512, 256]", getitem_7: "b8[1, 512, 256]", mul_3: "f32[1, 512, 256]", view_20: "f32[512, 256]", addmm_5: "f32[512, 1024]", view_22: "f32[512, 1024]", getitem_11: "b8[1, 512, 256]", mul_8: "f32[1, 512, 256]", view_24: "f32[512, 256]", getitem_147: "b8[1, 4, 512, 512]", permute_default_61: "f32[4, 512, 512]", permute_default_62: "f32[4, 64, 512]", alias_default_21: "f32[1, 4, 512, 512]", permute_default_63: "f32[4, 64, 512]", permute_default_64: "f32[4, 512, 64]", view_40: "f32[512, 256]", getitem_17: "b8[1, 512, 256]", mul_10: "f32[1, 512, 256]", view_42: "f32[512, 256]", addmm_11: "f32[512, 1024]", view_44: "f32[512, 1024]", getitem_21: "b8[1, 512, 256]", mul_15: "f32[1, 512, 256]", view_46: "f32[512, 256]", getitem_145: "b8[1, 4, 512, 512]", permute_default_55: "f32[4, 512, 512]", permute_default_56: "f32[4, 64, 512]", alias_default_19: "f32[1, 4, 512, 512]", permute_default_57: "f32[4, 64, 512]", permute_default_58: "f32[4, 512, 64]", view_62: "f32[512, 256]", getitem_27: "b8[1, 512, 256]", mul_17: "f32[1, 512, 256]", view_64: "f32[512, 256]", addmm_17: "f32[512, 1024]", view_66: "f32[512, 1024]", getitem_31: "b8[1, 512, 256]", mul_22: "f32[1, 512, 256]", view_68: "f32[512, 256]", getitem_143: "b8[1, 4, 512, 512]", permute_default_49: "f32[4, 512, 512]", permute_default_50: "f32[4, 64, 512]", alias_default_17: "f32[1, 4, 512, 512]", permute_default_51: "f32[4, 64, 512]", permute_default_52: "f32[4, 512, 64]", view_84: "f32[512, 256]", getitem_37: "b8[1, 512, 256]", mul_24: "f32[1, 512, 256]", view_86: "f32[512, 256]", addmm_23: "f32[512, 1024]", view_88: "f32[512, 1024]", getitem_41: "b8[1, 512, 256]", mul_29: "f32[1, 512, 256]", view_90: "f32[512, 256]", getitem_141: "b8[1, 4, 512, 512]", permute_default_43: "f32[4, 512, 512]", permute_default_44: "f32[4, 64, 512]", alias_default_15: "f32[1, 4, 512, 512]", permute_default_45: "f32[4, 64, 512]", permute_default_46: "f32[4, 512, 64]", view_106: "f32[512, 256]", getitem_47: "b8[1, 512, 256]", mul_31: "f32[1, 512, 256]", view_108: "f32[512, 256]", addmm_29: "f32[512, 1024]", view_110: "f32[512, 1024]", getitem_51: "b8[1, 512, 256]", mul_36: "f32[1, 512, 256]", view_112: "f32[512, 256]", getitem_139: "b8[1, 4, 512, 512]", permute_default_37: "f32[4, 512, 512]", permute_default_38: "f32[4, 64, 512]", alias_default_13: "f32[1, 4, 512, 512]", permute_default_39: "f32[4, 64, 512]", permute_default_40: "f32[4, 512, 64]", view_128: "f32[512, 256]", getitem_57: "b8[1, 512, 256]", mul_38: "f32[1, 512, 256]", view_130: "f32[512, 256]", addmm_35: "f32[512, 1024]", view_132: "f32[512, 1024]", getitem_61: "b8[1, 512, 256]", mul_43: "f32[1, 512, 256]", view_134: "f32[512, 256]", getitem_137: "b8[1, 4, 512, 512]", permute_default_31: "f32[4, 512, 512]", permute_default_32: "f32[4, 64, 512]", alias_default_11: "f32[1, 4, 512, 512]", permute_default_33: "f32[4, 64, 512]", permute_default_34: "f32[4, 512, 64]", view_150: "f32[512, 256]", getitem_67: "b8[1, 512, 256]", mul_45: "f32[1, 512, 256]", view_152: "f32[512, 256]", addmm_41: "f32[512, 1024]", view_154: "f32[512, 1024]", getitem_71: "b8[1, 512, 256]", mul_50: "f32[1, 512, 256]", view_156: "f32[512, 256]", getitem_135: "b8[1, 4, 512, 512]", permute_default_25: "f32[4, 512, 512]", permute_default_26: "f32[4, 64, 512]", alias_default_9: "f32[1, 4, 512, 512]", permute_default_27: "f32[4, 64, 512]", permute_default_28: "f32[4, 512, 64]", view_172: "f32[512, 256]", getitem_77: "b8[1, 512, 256]", mul_52: "f32[1, 512, 256]", view_174: "f32[512, 256]", addmm_47: "f32[512, 1024]", view_176: "f32[512, 1024]", getitem_81: "b8[1, 512, 256]", mul_57: "f32[1, 512, 256]", view_178: "f32[512, 256]", getitem_133: "b8[1, 4, 512, 512]", permute_default_19: "f32[4, 512, 512]", permute_default_20: "f32[4, 64, 512]", alias_default_7: "f32[1, 4, 512, 512]", permute_default_21: "f32[4, 64, 512]", permute_default_22: "f32[4, 512, 64]", view_194: "f32[512, 256]", getitem_87: "b8[1, 512, 256]", mul_59: "f32[1, 512, 256]", view_196: "f32[512, 256]", addmm_53: "f32[512, 1024]", view_198: "f32[512, 1024]", getitem_91: "b8[1, 512, 256]", mul_64: "f32[1, 512, 256]", view_200: "f32[512, 256]", getitem_131: "b8[1, 4, 512, 512]", permute_default_13: "f32[4, 512, 512]", permute_default_14: "f32[4, 64, 512]", alias_default_5: "f32[1, 4, 512, 512]", permute_default_15: "f32[4, 64, 512]", permute_default_16: "f32[4, 512, 64]", view_216: "f32[512, 256]", getitem_97: "b8[1, 512, 256]", mul_66: "f32[1, 512, 256]", view_218: "f32[512, 256]", addmm_59: "f32[512, 1024]", view_220: "f32[512, 1024]", getitem_101: "b8[1, 512, 256]", mul_71: "f32[1, 512, 256]", view_222: "f32[512, 256]", getitem_129: "b8[1, 4, 512, 512]", permute_default_7: "f32[4, 512, 512]", permute_default_8: "f32[4, 64, 512]", alias_default_3: "f32[1, 4, 512, 512]", permute_default_9: "f32[4, 64, 512]", permute_default_10: "f32[4, 512, 64]", view_238: "f32[512, 256]", getitem_107: "b8[1, 512, 256]", mul_73: "f32[1, 512, 256]", view_240: "f32[512, 256]", addmm_65: "f32[512, 1024]", view_242: "f32[512, 1024]", getitem_111: "b8[1, 512, 256]", mul_78: "f32[1, 512, 256]", view_244: "f32[512, 256]", getitem_127: "b8[1, 4, 512, 512]", permute_default_1: "f32[4, 512, 512]", permute_default_2: "f32[4, 64, 512]", alias_default_1: "f32[1, 4, 512, 512]", permute_default_3: "f32[4, 64, 512]", permute_default_4: "f32[4, 512, 64]", view_260: "f32[512, 256]", getitem_117: "b8[1, 512, 256]", mul_80: "f32[1, 512, 256]", view_262: "f32[512, 256]", addmm_71: "f32[512, 1024]", view_264: "f32[512, 1024]", getitem_121: "b8[1, 512, 256]", mul_85: "f32[1, 512, 256]", view_266: "f32[512, 256]", addmm_73: "f32[512, 128]", mul_90: "f32[1, 512, 128]", view_268: "f32[512, 128]", sub_40: "f32[511, 30522]", convert_element_type: "f32[]", ne_3: "b8[511, 1]", where_2: "i64[511, 1]", permute_135: "f32[30522, 128]", div_26: "f32[1, 512, 1]", permute_139: "f32[128, 256]", div_27: "f32[1, 512, 1]", permute_143: "f32[256, 1024]", permute_147: "f32[1024, 256]", div_28: "f32[1, 512, 1]", permute_151: "f32[256, 256]", permute_163: "f32[256, 256]", permute_168: "f32[256, 256]", permute_172: "f32[256, 256]", div_30: "f32[1, 512, 1]", permute_176: "f32[256, 1024]", permute_180: "f32[1024, 256]", div_31: "f32[1, 512, 1]", permute_184: "f32[256, 256]", permute_196: "f32[256, 256]", permute_201: "f32[256, 256]", permute_205: "f32[256, 256]", div_33: "f32[1, 512, 1]", permute_209: "f32[256, 1024]", permute_213: "f32[1024, 256]", div_34: "f32[1, 512, 1]", permute_217: "f32[256, 256]", permute_229: "f32[256, 256]", permute_234: "f32[256, 256]", permute_238: "f32[256, 256]", div_36: "f32[1, 512, 1]", permute_242: "f32[256, 1024]", permute_246: "f32[1024, 256]", div_37: "f32[1, 512, 1]", permute_250: "f32[256, 256]", permute_262: "f32[256, 256]", permute_267: "f32[256, 256]", permute_271: "f32[256, 256]", div_39: "f32[1, 512, 1]", permute_275: "f32[256, 1024]", permute_279: "f32[1024, 256]", div_40: "f32[1, 512, 1]", permute_283: "f32[256, 256]", permute_295: "f32[256, 256]", permute_300: "f32[256, 256]", permute_304: "f32[256, 256]", div_42: "f32[1, 512, 1]", permute_308: "f32[256, 1024]", permute_312: "f32[1024, 256]", div_43: "f32[1, 512, 1]", permute_316: "f32[256, 256]", permute_328: "f32[256, 256]", permute_333: "f32[256, 256]", permute_337: "f32[256, 256]", div_45: "f32[1, 512, 1]", permute_341: "f32[256, 1024]", permute_345: "f32[1024, 256]", div_46: "f32[1, 512, 1]", permute_349: "f32[256, 256]", permute_361: "f32[256, 256]", permute_366: "f32[256, 256]", permute_370: "f32[256, 256]", div_48: "f32[1, 512, 1]", permute_374: "f32[256, 1024]", permute_378: "f32[1024, 256]", div_49: "f32[1, 512, 1]", permute_382: "f32[256, 256]", permute_394: "f32[256, 256]", permute_399: "f32[256, 256]", permute_403: "f32[256, 256]", div_51: "f32[1, 512, 1]", permute_407: "f32[256, 1024]", permute_411: "f32[1024, 256]", div_52: "f32[1, 512, 1]", permute_415: "f32[256, 256]", permute_427: "f32[256, 256]", permute_432: "f32[256, 256]", permute_436: "f32[256, 256]", div_54: "f32[1, 512, 1]", permute_440: "f32[256, 1024]", permute_444: "f32[1024, 256]", div_55: "f32[1, 512, 1]", permute_448: "f32[256, 256]", permute_460: "f32[256, 256]", permute_465: "f32[256, 256]", permute_469: "f32[256, 256]", div_57: "f32[1, 512, 1]", permute_473: "f32[256, 1024]", permute_477: "f32[1024, 256]", div_58: "f32[1, 512, 1]", permute_481: "f32[256, 256]", permute_493: "f32[256, 256]", permute_498: "f32[256, 256]", permute_502: "f32[256, 256]", div_60: "f32[1, 512, 1]", permute_506: "f32[256, 1024]", permute_510: "f32[1024, 256]", div_61: "f32[1, 512, 1]", permute_514: "f32[256, 256]", permute_526: "f32[256, 256]", permute_531: "f32[256, 256]", permute_535: "f32[256, 256]", permute_539: "f32[256, 128]", div_63: "f32[1, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[1, 512, 30522]"):
    # No stacktrace found for following nodes
    convert_element_type_default_11: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_149, torch.float32);  getitem_149 = None
    mul_tensor_44: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, 1.1111111111111112);  convert_element_type_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_21: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_5, [1, 512, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, 0.7071067811865476)
    erf: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_8: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_10: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
    mul_tensor_40: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 1.1111111111111112);  convert_element_type_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_43: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_11, [1, 512, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_13: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_1: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_16: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_9: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_145, torch.float32);  getitem_145 = None
    mul_tensor_36: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_65: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_17, [1, 512, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, 0.7071067811865476)
    erf_2: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_24: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_8: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
    mul_tensor_32: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 1.1111111111111112);  convert_element_type_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_87: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_23, [1, 512, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_27: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, 0.7071067811865476)
    erf_3: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_27);  mul_27 = None
    add_32: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_7: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
    mul_tensor_28: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.1111111111111112);  convert_element_type_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_109: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_29, [1, 512, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_34: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476)
    erf_4: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_34);  mul_34 = None
    add_40: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_6: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_139, torch.float32);  getitem_139 = None
    mul_tensor_24: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_131: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_35, [1, 512, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_41: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.7071067811865476)
    erf_5: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_48: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_5: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
    mul_tensor_20: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.1111111111111112);  convert_element_type_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_153: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_41, [1, 512, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_48: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, 0.7071067811865476)
    erf_6: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_56: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_4: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
    mul_tensor_16: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_175: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_47, [1, 512, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_55: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, 0.7071067811865476)
    erf_7: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_64: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_3: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_133, torch.float32);  getitem_133 = None
    mul_tensor_12: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1.1111111111111112);  convert_element_type_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_197: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_53, [1, 512, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, 0.7071067811865476)
    erf_8: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
    add_72: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_2: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
    mul_tensor_8: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_219: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_59, [1, 512, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_69: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, 0.7071067811865476)
    erf_9: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_80: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default_1: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
    mul_tensor_4: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_241: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_65, [1, 512, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, 0.7071067811865476)
    erf_10: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_88: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # No stacktrace found for following nodes
    convert_element_type_default: "f32[1, 4, 512, 512]" = torch.ops.prims.convert_element_type.default(getitem_127, torch.float32);  getitem_127 = None
    mul_tensor: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_263: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(addmm_71, [1, 512, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_83: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, 0.7071067811865476)
    erf_11: "f32[1, 512, 1024]" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
    add_96: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_267: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(addmm_73, [1, 512, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_88: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, 0.7071067811865476)
    erf_12: "f32[1, 512, 128]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_100: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1648, code: lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    div_25: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None
    full_default_4: "f32[511, 30522]" = torch.ops.aten.full.default([511, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    scatter: "f32[511, 30522]" = torch.ops.aten.scatter.value(full_default_4, 1, where_2, -1.0);  full_default_4 = where_2 = None
    where_3: "f32[511, 1]" = torch.ops.aten.where.self(ne_3, div_25, full_default_2);  ne_3 = div_25 = None
    mul_92: "f32[511, 30522]" = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
    exp_13: "f32[511, 30522]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_16: "f32[511, 1]" = torch.ops.aten.sum.dim_IntList(mul_92, [1], True)
    mul_93: "f32[511, 30522]" = torch.ops.aten.mul.Tensor(exp_13, sum_16);  exp_13 = sum_16 = None
    sub_41: "f32[511, 30522]" = torch.ops.aten.sub.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    view_272: "f32[1, 511, 30522]" = torch.ops.aten.reshape.default(sub_41, [1, 511, 30522]);  sub_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    full_default_6: "f32[1, 511, 30522]" = torch.ops.aten.full.default([1, 511, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    full_default_7: "f32[1, 512, 30522]" = torch.ops.aten.full.default([1, 512, 30522], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[1, 512, 30522]" = torch.ops.aten.slice_scatter.default(full_default_7, view_272, 1, 0, -1);  full_default_7 = view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1645, code: shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
    add_103: "f32[1, 512, 30522]" = torch.ops.aten.add.Tensor(tangents_2, slice_scatter_1);  tangents_2 = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:1640, code: prediction_scores = self.generator_lm_head(self.generator_predictions(sequence_output))
    view_273: "f32[512, 30522]" = torch.ops.aten.reshape.default(add_103, [512, 30522]);  add_103 = None
    mm: "f32[512, 128]" = torch.ops.aten.mm.default(view_273, permute_135);  permute_135 = None
    permute_136: "f32[30522, 512]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_1: "f32[30522, 128]" = torch.ops.aten.mm.default(permute_136, view_268);  permute_136 = view_268 = None
    permute_137: "f32[128, 30522]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 30522]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[30522]" = torch.ops.aten.reshape.default(sum_17, [30522]);  sum_17 = None
    permute_138: "f32[30522, 128]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    view_275: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(mm, [1, 512, 128]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:662, code: hidden_states = self.LayerNorm(hidden_states)
    mul_95: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, primals_202);  primals_202 = None
    mul_96: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_95, 128)
    sum_18: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
    mul_97: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_95, mul_90);  mul_95 = None
    sum_19: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
    mul_98: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_90, sum_19);  sum_19 = None
    sub_43: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_96, sum_18);  mul_96 = sum_18 = None
    sub_44: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_43, mul_98);  sub_43 = mul_98 = None
    mul_99: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_26, sub_44);  div_26 = sub_44 = None
    mul_100: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_275, mul_90);  mul_90 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1]);  view_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_102: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(add_100, 0.5);  add_100 = None
    mul_103: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, view_267)
    mul_104: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_103, -0.5);  mul_103 = None
    exp_14: "f32[1, 512, 128]" = torch.ops.aten.exp.default(mul_104);  mul_104 = None
    mul_105: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_106: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_267, mul_105);  view_267 = mul_105 = None
    add_105: "f32[1, 512, 128]" = torch.ops.aten.add.Tensor(mul_102, mul_106);  mul_102 = mul_106 = None
    mul_107: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_99, add_105);  mul_99 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:660, code: hidden_states = self.dense(generator_hidden_states)
    view_276: "f32[512, 128]" = torch.ops.aten.reshape.default(mul_107, [512, 128]);  mul_107 = None
    mm_2: "f32[512, 256]" = torch.ops.aten.mm.default(view_276, permute_139);  permute_139 = None
    permute_140: "f32[128, 512]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_3: "f32[128, 256]" = torch.ops.aten.mm.default(permute_140, view_266);  permute_140 = view_266 = None
    permute_141: "f32[256, 128]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[128]" = torch.ops.aten.reshape.default(sum_22, [128]);  sum_22 = None
    permute_142: "f32[128, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    view_278: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_2, [1, 512, 256]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_109: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, primals_198);  primals_198 = None
    mul_110: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_109, 256)
    sum_23: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_109, mul_85);  mul_109 = None
    sum_24: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_85, sum_24);  sum_24 = None
    sub_46: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_110, sum_23);  mul_110 = sum_23 = None
    sub_47: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_46, mul_112);  sub_46 = mul_112 = None
    mul_113: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_47);  div_27 = sub_47 = None
    mul_114: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(view_278, mul_85);  mul_85 = None
    sum_25: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_26: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_1: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_121, torch.float32);  getitem_121 = None
    mul_115: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
    mul_116: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_113, mul_115);  mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_279: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_116, [512, 256]);  mul_116 = None
    mm_4: "f32[512, 1024]" = torch.ops.aten.mm.default(view_279, permute_143);  permute_143 = None
    permute_144: "f32[256, 512]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_5: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_144, view_264);  permute_144 = view_264 = None
    permute_145: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_27: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[256]" = torch.ops.aten.reshape.default(sum_27, [256]);  sum_27 = None
    permute_146: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
    view_281: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 512, 1024]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_119: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, view_263)
    mul_120: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_119, -0.5);  mul_119 = None
    exp_15: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_120);  mul_120 = None
    mul_121: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_122: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_263, mul_121);  view_263 = mul_121 = None
    add_107: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_118, mul_122);  mul_118 = mul_122 = None
    mul_123: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_281, add_107);  view_281 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_282: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_123, [512, 1024]);  mul_123 = None
    mm_6: "f32[512, 256]" = torch.ops.aten.mm.default(view_282, permute_147);  permute_147 = None
    permute_148: "f32[1024, 512]" = torch.ops.aten.permute.default(view_282, [1, 0])
    mm_7: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_148, view_262);  permute_148 = view_262 = None
    permute_149: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_28: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_282, [0], True);  view_282 = None
    view_283: "f32[1024]" = torch.ops.aten.reshape.default(sum_28, [1024]);  sum_28 = None
    permute_150: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    view_284: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_6, [1, 512, 256]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_108: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_113, view_284);  mul_113 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_125: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, primals_192);  primals_192 = None
    mul_126: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, 256)
    sum_29: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True)
    mul_127: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_125, mul_80);  mul_125 = None
    sum_30: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True);  mul_127 = None
    mul_128: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_80, sum_30);  sum_30 = None
    sub_49: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_126, sum_29);  mul_126 = sum_29 = None
    sub_50: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_49, mul_128);  sub_49 = mul_128 = None
    mul_129: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_28, sub_50);  div_28 = sub_50 = None
    mul_130: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_108, mul_80);  mul_80 = None
    sum_31: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1]);  mul_130 = None
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_2: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_131: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
    mul_132: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_129, mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_285: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_132, [512, 256]);  mul_132 = None
    mm_8: "f32[512, 256]" = torch.ops.aten.mm.default(view_285, permute_151);  permute_151 = None
    permute_152: "f32[256, 512]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_9: "f32[256, 256]" = torch.ops.aten.mm.default(permute_152, view_260);  permute_152 = view_260 = None
    permute_153: "f32[256, 256]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_33: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[256]" = torch.ops.aten.reshape.default(sum_33, [256]);  sum_33 = None
    permute_154: "f32[256, 256]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_287: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_8, [1, 512, 256]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_288: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_287, [1, 512, 4, 64]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_155: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
    
    # No stacktrace found for following nodes
    view_default_6: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_155, [4, 512, 64]);  permute_155 = None
    bmm_default_2: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_1, view_default_6);  permute_default_1 = None
    view_default_7: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_2, [1, 4, 512, 64]);  bmm_default_2 = None
    bmm_default_3: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
    view_default_8: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_3, [1, 4, 512, 512]);  bmm_default_3 = None
    mul_tensor_1: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_8, mul_tensor);  view_default_8 = mul_tensor = None
    mul_tensor_2: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_1, alias_default_1);  mul_tensor_1 = None
    sum_dim_int_list_1: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True)
    mul_tensor_3: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_1, sum_dim_int_list_1);  alias_default_1 = sum_dim_int_list_1 = None
    sub_tensor_1: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3);  mul_tensor_2 = mul_tensor_3 = None
    view_default_9: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_1, [4, 512, 512]);  sub_tensor_1 = None
    bmm_default_4: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_3, view_default_9);  permute_default_3 = None
    view_default_10: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_4, [1, 4, 64, 512]);  bmm_default_4 = None
    mul_scalar_2: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_10, 0.3535533905932738);  view_default_10 = None
    permute_default_5: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_2, [0, 1, 3, 2]);  mul_scalar_2 = None
    bmm_default_5: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_9, permute_default_4);  view_default_9 = permute_default_4 = None
    view_default_11: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_5, [1, 4, 512, 64]);  bmm_default_5 = None
    mul_scalar_3: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_11, 0.3535533905932738);  view_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_161: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_3, [0, 2, 1, 3]);  mul_scalar_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_15: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_295: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_15, [1, 512, 256]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_162: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_16: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_296: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_16, [1, 512, 256]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_297: "f32[512, 256]" = torch.ops.aten.reshape.default(view_296, [512, 256]);  view_296 = None
    mm_10: "f32[512, 256]" = torch.ops.aten.mm.default(view_297, permute_163);  permute_163 = None
    permute_164: "f32[256, 512]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_11: "f32[256, 256]" = torch.ops.aten.mm.default(permute_164, view_244);  permute_164 = None
    permute_165: "f32[256, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[256]" = torch.ops.aten.reshape.default(sum_35, [256]);  sum_35 = None
    permute_166: "f32[256, 256]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_299: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_10, [1, 512, 256]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_109: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_129, view_299);  mul_129 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_167: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_5, [0, 2, 1, 3]);  permute_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_300: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_167, [1, 512, 256]);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_301: "f32[512, 256]" = torch.ops.aten.reshape.default(view_300, [512, 256]);  view_300 = None
    mm_12: "f32[512, 256]" = torch.ops.aten.mm.default(view_301, permute_168);  permute_168 = None
    permute_169: "f32[256, 512]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_13: "f32[256, 256]" = torch.ops.aten.mm.default(permute_169, view_244);  permute_169 = None
    permute_170: "f32[256, 256]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[256]" = torch.ops.aten.reshape.default(sum_36, [256]);  sum_36 = None
    permute_171: "f32[256, 256]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_303: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_12, [1, 512, 256]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_110: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_109, view_303);  add_109 = view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_304: "f32[512, 256]" = torch.ops.aten.reshape.default(view_295, [512, 256]);  view_295 = None
    mm_14: "f32[512, 256]" = torch.ops.aten.mm.default(view_304, permute_172);  permute_172 = None
    permute_173: "f32[256, 512]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_15: "f32[256, 256]" = torch.ops.aten.mm.default(permute_173, view_244);  permute_173 = view_244 = None
    permute_174: "f32[256, 256]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_37: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[256]" = torch.ops.aten.reshape.default(sum_37, [256]);  sum_37 = None
    permute_175: "f32[256, 256]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_306: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_14, [1, 512, 256]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_111: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_110, view_306);  add_110 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_138: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, primals_182);  primals_182 = None
    mul_139: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_138, 256)
    sum_38: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
    mul_140: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_138, mul_78);  mul_138 = None
    sum_39: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
    mul_141: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_78, sum_39);  sum_39 = None
    sub_53: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_139, sum_38);  mul_139 = sum_38 = None
    sub_54: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_53, mul_141);  sub_53 = mul_141 = None
    mul_142: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_54);  div_30 = sub_54 = None
    mul_143: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_111, mul_78);  mul_78 = None
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_4: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_144: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
    mul_145: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_142, mul_144);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_307: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_145, [512, 256]);  mul_145 = None
    mm_16: "f32[512, 1024]" = torch.ops.aten.mm.default(view_307, permute_176);  permute_176 = None
    permute_177: "f32[256, 512]" = torch.ops.aten.permute.default(view_307, [1, 0])
    mm_17: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_177, view_242);  permute_177 = view_242 = None
    permute_178: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_42: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0], True);  view_307 = None
    view_308: "f32[256]" = torch.ops.aten.reshape.default(sum_42, [256]);  sum_42 = None
    permute_179: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_309: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_16, [1, 512, 1024]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_147: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
    mul_148: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, view_241)
    mul_149: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_148, -0.5);  mul_148 = None
    exp_16: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_149);  mul_149 = None
    mul_150: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_151: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_241, mul_150);  view_241 = mul_150 = None
    add_113: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_147, mul_151);  mul_147 = mul_151 = None
    mul_152: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_309, add_113);  view_309 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_310: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_152, [512, 1024]);  mul_152 = None
    mm_18: "f32[512, 256]" = torch.ops.aten.mm.default(view_310, permute_180);  permute_180 = None
    permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(view_310, [1, 0])
    mm_19: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_181, view_240);  permute_181 = view_240 = None
    permute_182: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_43: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
    view_311: "f32[1024]" = torch.ops.aten.reshape.default(sum_43, [1024]);  sum_43 = None
    permute_183: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_312: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_18, [1, 512, 256]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_114: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_142, view_312);  mul_142 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_154: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, primals_176);  primals_176 = None
    mul_155: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, 256)
    sum_44: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True)
    mul_156: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_154, mul_73);  mul_154 = None
    sum_45: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True);  mul_156 = None
    mul_157: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_73, sum_45);  sum_45 = None
    sub_56: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_155, sum_44);  mul_155 = sum_44 = None
    sub_57: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_56, mul_157);  sub_56 = mul_157 = None
    mul_158: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_57);  div_31 = sub_57 = None
    mul_159: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_114, mul_73);  mul_73 = None
    sum_46: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_159, [0, 1]);  mul_159 = None
    sum_47: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_5: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_160: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
    mul_161: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_158, mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_313: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_161, [512, 256]);  mul_161 = None
    mm_20: "f32[512, 256]" = torch.ops.aten.mm.default(view_313, permute_184);  permute_184 = None
    permute_185: "f32[256, 512]" = torch.ops.aten.permute.default(view_313, [1, 0])
    mm_21: "f32[256, 256]" = torch.ops.aten.mm.default(permute_185, view_238);  permute_185 = view_238 = None
    permute_186: "f32[256, 256]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_48: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_313, [0], True);  view_313 = None
    view_314: "f32[256]" = torch.ops.aten.reshape.default(sum_48, [256]);  sum_48 = None
    permute_187: "f32[256, 256]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_315: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_20, [1, 512, 256]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_316: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_315, [1, 512, 4, 64]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_188: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # No stacktrace found for following nodes
    view_default_18: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_188, [4, 512, 64]);  permute_188 = None
    bmm_default_8: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_7, view_default_18);  permute_default_7 = None
    view_default_19: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_8, [1, 4, 512, 64]);  bmm_default_8 = None
    bmm_default_9: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_18, permute_default_8);  view_default_18 = permute_default_8 = None
    view_default_20: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_9, [1, 4, 512, 512]);  bmm_default_9 = None
    mul_tensor_5: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_20, mul_tensor_4);  view_default_20 = mul_tensor_4 = None
    mul_tensor_6: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_5, alias_default_3);  mul_tensor_5 = None
    sum_dim_int_list_3: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [-1], True)
    mul_tensor_7: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_3, sum_dim_int_list_3);  alias_default_3 = sum_dim_int_list_3 = None
    sub_tensor_3: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_6, mul_tensor_7);  mul_tensor_6 = mul_tensor_7 = None
    view_default_21: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_3, [4, 512, 512]);  sub_tensor_3 = None
    bmm_default_10: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_9, view_default_21);  permute_default_9 = None
    view_default_22: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_10, [1, 4, 64, 512]);  bmm_default_10 = None
    mul_scalar_6: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_22, 0.3535533905932738);  view_default_22 = None
    permute_default_11: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_6, [0, 1, 3, 2]);  mul_scalar_6 = None
    bmm_default_11: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_21, permute_default_10);  view_default_21 = permute_default_10 = None
    view_default_23: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_11, [1, 4, 512, 64]);  bmm_default_11 = None
    mul_scalar_7: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_23, 0.3535533905932738);  view_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_194: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_7, [0, 2, 1, 3]);  mul_scalar_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_20: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_323: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_20, [1, 512, 256]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_195: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_19, [0, 2, 1, 3]);  view_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_21: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    view_324: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_21, [1, 512, 256]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_325: "f32[512, 256]" = torch.ops.aten.reshape.default(view_324, [512, 256]);  view_324 = None
    mm_22: "f32[512, 256]" = torch.ops.aten.mm.default(view_325, permute_196);  permute_196 = None
    permute_197: "f32[256, 512]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_197, view_222);  permute_197 = None
    permute_198: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[256]" = torch.ops.aten.reshape.default(sum_50, [256]);  sum_50 = None
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
    view_327: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_22, [1, 512, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_115: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_158, view_327);  mul_158 = view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_200: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_11, [0, 2, 1, 3]);  permute_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_328: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_200, [1, 512, 256]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_329: "f32[512, 256]" = torch.ops.aten.reshape.default(view_328, [512, 256]);  view_328 = None
    mm_24: "f32[512, 256]" = torch.ops.aten.mm.default(view_329, permute_201);  permute_201 = None
    permute_202: "f32[256, 512]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_202, view_222);  permute_202 = None
    permute_203: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_51: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[256]" = torch.ops.aten.reshape.default(sum_51, [256]);  sum_51 = None
    permute_204: "f32[256, 256]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    view_331: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_24, [1, 512, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_116: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_115, view_331);  add_115 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_332: "f32[512, 256]" = torch.ops.aten.reshape.default(view_323, [512, 256]);  view_323 = None
    mm_26: "f32[512, 256]" = torch.ops.aten.mm.default(view_332, permute_205);  permute_205 = None
    permute_206: "f32[256, 512]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_27: "f32[256, 256]" = torch.ops.aten.mm.default(permute_206, view_222);  permute_206 = view_222 = None
    permute_207: "f32[256, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_52: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[256]" = torch.ops.aten.reshape.default(sum_52, [256]);  sum_52 = None
    permute_208: "f32[256, 256]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_334: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_26, [1, 512, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_117: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_116, view_334);  add_116 = view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_167: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, primals_166);  primals_166 = None
    mul_168: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, 256)
    sum_53: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_167, mul_71);  mul_167 = None
    sum_54: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_71, sum_54);  sum_54 = None
    sub_60: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_168, sum_53);  mul_168 = sum_53 = None
    sub_61: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_60, mul_170);  sub_60 = mul_170 = None
    mul_171: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_61);  div_33 = sub_61 = None
    mul_172: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_117, mul_71);  mul_71 = None
    sum_55: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_117, [0, 1]);  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_7: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_173: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
    mul_174: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_171, mul_173);  mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_335: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_174, [512, 256]);  mul_174 = None
    mm_28: "f32[512, 1024]" = torch.ops.aten.mm.default(view_335, permute_209);  permute_209 = None
    permute_210: "f32[256, 512]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_29: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_210, view_220);  permute_210 = view_220 = None
    permute_211: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_57: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[256]" = torch.ops.aten.reshape.default(sum_57, [256]);  sum_57 = None
    permute_212: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_337: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_28, [1, 512, 1024]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_176: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
    mul_177: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, view_219)
    mul_178: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_177, -0.5);  mul_177 = None
    exp_17: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_178);  mul_178 = None
    mul_179: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_180: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_219, mul_179);  view_219 = mul_179 = None
    add_119: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_176, mul_180);  mul_176 = mul_180 = None
    mul_181: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_337, add_119);  view_337 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_338: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_181, [512, 1024]);  mul_181 = None
    mm_30: "f32[512, 256]" = torch.ops.aten.mm.default(view_338, permute_213);  permute_213 = None
    permute_214: "f32[1024, 512]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_31: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_214, view_218);  permute_214 = view_218 = None
    permute_215: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_58: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[1024]" = torch.ops.aten.reshape.default(sum_58, [1024]);  sum_58 = None
    permute_216: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_340: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_30, [1, 512, 256]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_120: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_171, view_340);  mul_171 = view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_183: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, primals_160);  primals_160 = None
    mul_184: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, 256)
    sum_59: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True)
    mul_185: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_183, mul_66);  mul_183 = None
    sum_60: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True);  mul_185 = None
    mul_186: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_66, sum_60);  sum_60 = None
    sub_63: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_184, sum_59);  mul_184 = sum_59 = None
    sub_64: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_63, mul_186);  sub_63 = mul_186 = None
    mul_187: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_64);  div_34 = sub_64 = None
    mul_188: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_120, mul_66);  mul_66 = None
    sum_61: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 1]);  mul_188 = None
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_8: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_189: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
    mul_190: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_187, mul_189);  mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_341: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_190, [512, 256]);  mul_190 = None
    mm_32: "f32[512, 256]" = torch.ops.aten.mm.default(view_341, permute_217);  permute_217 = None
    permute_218: "f32[256, 512]" = torch.ops.aten.permute.default(view_341, [1, 0])
    mm_33: "f32[256, 256]" = torch.ops.aten.mm.default(permute_218, view_216);  permute_218 = view_216 = None
    permute_219: "f32[256, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_63: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_341, [0], True);  view_341 = None
    view_342: "f32[256]" = torch.ops.aten.reshape.default(sum_63, [256]);  sum_63 = None
    permute_220: "f32[256, 256]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_343: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_32, [1, 512, 256]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_344: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_343, [1, 512, 4, 64]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_221: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
    
    # No stacktrace found for following nodes
    view_default_30: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_221, [4, 512, 64]);  permute_221 = None
    bmm_default_14: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_13, view_default_30);  permute_default_13 = None
    view_default_31: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_14, [1, 4, 512, 64]);  bmm_default_14 = None
    bmm_default_15: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_30, permute_default_14);  view_default_30 = permute_default_14 = None
    view_default_32: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_15, [1, 4, 512, 512]);  bmm_default_15 = None
    mul_tensor_9: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_32, mul_tensor_8);  view_default_32 = mul_tensor_8 = None
    mul_tensor_10: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_9, alias_default_5);  mul_tensor_9 = None
    sum_dim_int_list_5: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [-1], True)
    mul_tensor_11: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_5, sum_dim_int_list_5);  alias_default_5 = sum_dim_int_list_5 = None
    sub_tensor_5: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
    view_default_33: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_5, [4, 512, 512]);  sub_tensor_5 = None
    bmm_default_16: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_15, view_default_33);  permute_default_15 = None
    view_default_34: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_16, [1, 4, 64, 512]);  bmm_default_16 = None
    mul_scalar_10: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_34, 0.3535533905932738);  view_default_34 = None
    permute_default_17: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_10, [0, 1, 3, 2]);  mul_scalar_10 = None
    bmm_default_17: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_33, permute_default_16);  view_default_33 = permute_default_16 = None
    view_default_35: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_17, [1, 4, 512, 64]);  bmm_default_17 = None
    mul_scalar_11: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_35, 0.3535533905932738);  view_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_227: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_11, [0, 2, 1, 3]);  mul_scalar_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_25: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_351: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_25, [1, 512, 256]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_228: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_31, [0, 2, 1, 3]);  view_default_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_26: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_352: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_26, [1, 512, 256]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_353: "f32[512, 256]" = torch.ops.aten.reshape.default(view_352, [512, 256]);  view_352 = None
    mm_34: "f32[512, 256]" = torch.ops.aten.mm.default(view_353, permute_229);  permute_229 = None
    permute_230: "f32[256, 512]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_35: "f32[256, 256]" = torch.ops.aten.mm.default(permute_230, view_200);  permute_230 = None
    permute_231: "f32[256, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_65: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.reshape.default(sum_65, [256]);  sum_65 = None
    permute_232: "f32[256, 256]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
    view_355: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_34, [1, 512, 256]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_121: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_187, view_355);  mul_187 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_233: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_17, [0, 2, 1, 3]);  permute_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_356: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_233, [1, 512, 256]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_357: "f32[512, 256]" = torch.ops.aten.reshape.default(view_356, [512, 256]);  view_356 = None
    mm_36: "f32[512, 256]" = torch.ops.aten.mm.default(view_357, permute_234);  permute_234 = None
    permute_235: "f32[256, 512]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_37: "f32[256, 256]" = torch.ops.aten.mm.default(permute_235, view_200);  permute_235 = None
    permute_236: "f32[256, 256]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_66: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
    view_358: "f32[256]" = torch.ops.aten.reshape.default(sum_66, [256]);  sum_66 = None
    permute_237: "f32[256, 256]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_359: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_36, [1, 512, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_122: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_121, view_359);  add_121 = view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_360: "f32[512, 256]" = torch.ops.aten.reshape.default(view_351, [512, 256]);  view_351 = None
    mm_38: "f32[512, 256]" = torch.ops.aten.mm.default(view_360, permute_238);  permute_238 = None
    permute_239: "f32[256, 512]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_39: "f32[256, 256]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[256, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_67: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
    view_361: "f32[256]" = torch.ops.aten.reshape.default(sum_67, [256]);  sum_67 = None
    permute_241: "f32[256, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_362: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_38, [1, 512, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_123: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_122, view_362);  add_122 = view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_196: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, primals_150);  primals_150 = None
    mul_197: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_196, 256)
    sum_68: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True)
    mul_198: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_196, mul_64);  mul_196 = None
    sum_69: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True);  mul_198 = None
    mul_199: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_64, sum_69);  sum_69 = None
    sub_67: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_197, sum_68);  mul_197 = sum_68 = None
    sub_68: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_67, mul_199);  sub_67 = mul_199 = None
    mul_200: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_36, sub_68);  div_36 = sub_68 = None
    mul_201: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_123, mul_64);  mul_64 = None
    sum_70: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_201, [0, 1]);  mul_201 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1]);  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_10: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_202: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
    mul_203: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_200, mul_202);  mul_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_363: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_203, [512, 256]);  mul_203 = None
    mm_40: "f32[512, 1024]" = torch.ops.aten.mm.default(view_363, permute_242);  permute_242 = None
    permute_243: "f32[256, 512]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_41: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_72: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[256]" = torch.ops.aten.reshape.default(sum_72, [256]);  sum_72 = None
    permute_245: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_365: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_40, [1, 512, 1024]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_205: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
    mul_206: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, view_197)
    mul_207: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_206, -0.5);  mul_206 = None
    exp_18: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_207);  mul_207 = None
    mul_208: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_209: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_197, mul_208);  view_197 = mul_208 = None
    add_125: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_205, mul_209);  mul_205 = mul_209 = None
    mul_210: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_365, add_125);  view_365 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_366: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_210, [512, 1024]);  mul_210 = None
    mm_42: "f32[512, 256]" = torch.ops.aten.mm.default(view_366, permute_246);  permute_246 = None
    permute_247: "f32[1024, 512]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_43: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_247, view_196);  permute_247 = view_196 = None
    permute_248: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_73: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[1024]" = torch.ops.aten.reshape.default(sum_73, [1024]);  sum_73 = None
    permute_249: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_368: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_42, [1, 512, 256]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_126: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_200, view_368);  mul_200 = view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_212: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, primals_144);  primals_144 = None
    mul_213: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, 256)
    sum_74: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True)
    mul_214: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_212, mul_59);  mul_212 = None
    sum_75: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    mul_215: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_59, sum_75);  sum_75 = None
    sub_70: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_213, sum_74);  mul_213 = sum_74 = None
    sub_71: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_70, mul_215);  sub_70 = mul_215 = None
    mul_216: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_37, sub_71);  div_37 = sub_71 = None
    mul_217: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_126, mul_59);  mul_59 = None
    sum_76: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1]);  mul_217 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_11: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_218: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
    mul_219: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_216, mul_218);  mul_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_369: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_219, [512, 256]);  mul_219 = None
    mm_44: "f32[512, 256]" = torch.ops.aten.mm.default(view_369, permute_250);  permute_250 = None
    permute_251: "f32[256, 512]" = torch.ops.aten.permute.default(view_369, [1, 0])
    mm_45: "f32[256, 256]" = torch.ops.aten.mm.default(permute_251, view_194);  permute_251 = view_194 = None
    permute_252: "f32[256, 256]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_78: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
    view_370: "f32[256]" = torch.ops.aten.reshape.default(sum_78, [256]);  sum_78 = None
    permute_253: "f32[256, 256]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_371: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_44, [1, 512, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_372: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_371, [1, 512, 4, 64]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_254: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
    
    # No stacktrace found for following nodes
    view_default_42: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_254, [4, 512, 64]);  permute_254 = None
    bmm_default_20: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_19, view_default_42);  permute_default_19 = None
    view_default_43: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_20, [1, 4, 512, 64]);  bmm_default_20 = None
    bmm_default_21: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_42, permute_default_20);  view_default_42 = permute_default_20 = None
    view_default_44: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_21, [1, 4, 512, 512]);  bmm_default_21 = None
    mul_tensor_13: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_44, mul_tensor_12);  view_default_44 = mul_tensor_12 = None
    mul_tensor_14: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_13, alias_default_7);  mul_tensor_13 = None
    sum_dim_int_list_7: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [-1], True)
    mul_tensor_15: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_7, sum_dim_int_list_7);  alias_default_7 = sum_dim_int_list_7 = None
    sub_tensor_7: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_14, mul_tensor_15);  mul_tensor_14 = mul_tensor_15 = None
    view_default_45: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_7, [4, 512, 512]);  sub_tensor_7 = None
    bmm_default_22: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_21, view_default_45);  permute_default_21 = None
    view_default_46: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_22, [1, 4, 64, 512]);  bmm_default_22 = None
    mul_scalar_14: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_46, 0.3535533905932738);  view_default_46 = None
    permute_default_23: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_14, [0, 1, 3, 2]);  mul_scalar_14 = None
    bmm_default_23: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_45, permute_default_22);  view_default_45 = permute_default_22 = None
    view_default_47: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_23, [1, 4, 512, 64]);  bmm_default_23 = None
    mul_scalar_15: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_47, 0.3535533905932738);  view_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_260: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_15, [0, 2, 1, 3]);  mul_scalar_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_30: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_379: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_30, [1, 512, 256]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_261: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_43, [0, 2, 1, 3]);  view_default_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_31: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_380: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_31, [1, 512, 256]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_381: "f32[512, 256]" = torch.ops.aten.reshape.default(view_380, [512, 256]);  view_380 = None
    mm_46: "f32[512, 256]" = torch.ops.aten.mm.default(view_381, permute_262);  permute_262 = None
    permute_263: "f32[256, 512]" = torch.ops.aten.permute.default(view_381, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_263, view_178);  permute_263 = None
    permute_264: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_80: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_381, [0], True);  view_381 = None
    view_382: "f32[256]" = torch.ops.aten.reshape.default(sum_80, [256]);  sum_80 = None
    permute_265: "f32[256, 256]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_383: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_46, [1, 512, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_127: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_216, view_383);  mul_216 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_266: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_23, [0, 2, 1, 3]);  permute_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_384: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_266, [1, 512, 256]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_385: "f32[512, 256]" = torch.ops.aten.reshape.default(view_384, [512, 256]);  view_384 = None
    mm_48: "f32[512, 256]" = torch.ops.aten.mm.default(view_385, permute_267);  permute_267 = None
    permute_268: "f32[256, 512]" = torch.ops.aten.permute.default(view_385, [1, 0])
    mm_49: "f32[256, 256]" = torch.ops.aten.mm.default(permute_268, view_178);  permute_268 = None
    permute_269: "f32[256, 256]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_81: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
    view_386: "f32[256]" = torch.ops.aten.reshape.default(sum_81, [256]);  sum_81 = None
    permute_270: "f32[256, 256]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_387: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_48, [1, 512, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_128: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_127, view_387);  add_127 = view_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_388: "f32[512, 256]" = torch.ops.aten.reshape.default(view_379, [512, 256]);  view_379 = None
    mm_50: "f32[512, 256]" = torch.ops.aten.mm.default(view_388, permute_271);  permute_271 = None
    permute_272: "f32[256, 512]" = torch.ops.aten.permute.default(view_388, [1, 0])
    mm_51: "f32[256, 256]" = torch.ops.aten.mm.default(permute_272, view_178);  permute_272 = view_178 = None
    permute_273: "f32[256, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_82: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_388, [0], True);  view_388 = None
    view_389: "f32[256]" = torch.ops.aten.reshape.default(sum_82, [256]);  sum_82 = None
    permute_274: "f32[256, 256]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    view_390: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_50, [1, 512, 256]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_129: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_128, view_390);  add_128 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_225: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, primals_134);  primals_134 = None
    mul_226: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_225, 256)
    sum_83: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
    mul_227: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_225, mul_57);  mul_225 = None
    sum_84: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
    mul_228: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_57, sum_84);  sum_84 = None
    sub_74: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_226, sum_83);  mul_226 = sum_83 = None
    sub_75: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_74, mul_228);  sub_74 = mul_228 = None
    mul_229: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_75);  div_39 = sub_75 = None
    mul_230: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_129, mul_57);  mul_57 = None
    sum_85: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
    sum_86: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_129, [0, 1]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_13: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_231: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
    mul_232: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_229, mul_231);  mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_391: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_232, [512, 256]);  mul_232 = None
    mm_52: "f32[512, 1024]" = torch.ops.aten.mm.default(view_391, permute_275);  permute_275 = None
    permute_276: "f32[256, 512]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_53: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_276, view_176);  permute_276 = view_176 = None
    permute_277: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_87: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[256]" = torch.ops.aten.reshape.default(sum_87, [256]);  sum_87 = None
    permute_278: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
    view_393: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_52, [1, 512, 1024]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_234: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
    mul_235: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, view_175)
    mul_236: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
    exp_19: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
    mul_237: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_238: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_237);  view_175 = mul_237 = None
    add_131: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
    mul_239: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_393, add_131);  view_393 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_394: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_239, [512, 1024]);  mul_239 = None
    mm_54: "f32[512, 256]" = torch.ops.aten.mm.default(view_394, permute_279);  permute_279 = None
    permute_280: "f32[1024, 512]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_55: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_280, view_174);  permute_280 = view_174 = None
    permute_281: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_88: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[1024]" = torch.ops.aten.reshape.default(sum_88, [1024]);  sum_88 = None
    permute_282: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
    view_396: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_54, [1, 512, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_132: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_229, view_396);  mul_229 = view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_241: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, primals_128);  primals_128 = None
    mul_242: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, 256)
    sum_89: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_241, mul_52);  mul_241 = None
    sum_90: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_52, sum_90);  sum_90 = None
    sub_77: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_242, sum_89);  mul_242 = sum_89 = None
    sub_78: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_77, mul_244);  sub_77 = mul_244 = None
    mul_245: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_78);  div_40 = sub_78 = None
    mul_246: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_132, mul_52);  mul_52 = None
    sum_91: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_92: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_14: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_247: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
    mul_248: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_245, mul_247);  mul_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_397: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_248, [512, 256]);  mul_248 = None
    mm_56: "f32[512, 256]" = torch.ops.aten.mm.default(view_397, permute_283);  permute_283 = None
    permute_284: "f32[256, 512]" = torch.ops.aten.permute.default(view_397, [1, 0])
    mm_57: "f32[256, 256]" = torch.ops.aten.mm.default(permute_284, view_172);  permute_284 = view_172 = None
    permute_285: "f32[256, 256]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_93: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_397, [0], True);  view_397 = None
    view_398: "f32[256]" = torch.ops.aten.reshape.default(sum_93, [256]);  sum_93 = None
    permute_286: "f32[256, 256]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_399: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_56, [1, 512, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_400: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_399, [1, 512, 4, 64]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_287: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_400, [0, 2, 1, 3]);  view_400 = None
    
    # No stacktrace found for following nodes
    view_default_54: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_287, [4, 512, 64]);  permute_287 = None
    bmm_default_26: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_25, view_default_54);  permute_default_25 = None
    view_default_55: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_26, [1, 4, 512, 64]);  bmm_default_26 = None
    bmm_default_27: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_54, permute_default_26);  view_default_54 = permute_default_26 = None
    view_default_56: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_27, [1, 4, 512, 512]);  bmm_default_27 = None
    mul_tensor_17: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_56, mul_tensor_16);  view_default_56 = mul_tensor_16 = None
    mul_tensor_18: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_17, alias_default_9);  mul_tensor_17 = None
    sum_dim_int_list_9: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [-1], True)
    mul_tensor_19: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_9, sum_dim_int_list_9);  alias_default_9 = sum_dim_int_list_9 = None
    sub_tensor_9: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None
    view_default_57: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_9, [4, 512, 512]);  sub_tensor_9 = None
    bmm_default_28: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_27, view_default_57);  permute_default_27 = None
    view_default_58: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_28, [1, 4, 64, 512]);  bmm_default_28 = None
    mul_scalar_18: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_58, 0.3535533905932738);  view_default_58 = None
    permute_default_29: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_18, [0, 1, 3, 2]);  mul_scalar_18 = None
    bmm_default_29: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_57, permute_default_28);  view_default_57 = permute_default_28 = None
    view_default_59: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_29, [1, 4, 512, 64]);  bmm_default_29 = None
    mul_scalar_19: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_59, 0.3535533905932738);  view_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_293: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_19, [0, 2, 1, 3]);  mul_scalar_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_35: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    view_407: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_35, [1, 512, 256]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_294: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_55, [0, 2, 1, 3]);  view_default_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_36: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_408: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_36, [1, 512, 256]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_409: "f32[512, 256]" = torch.ops.aten.reshape.default(view_408, [512, 256]);  view_408 = None
    mm_58: "f32[512, 256]" = torch.ops.aten.mm.default(view_409, permute_295);  permute_295 = None
    permute_296: "f32[256, 512]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_59: "f32[256, 256]" = torch.ops.aten.mm.default(permute_296, view_156);  permute_296 = None
    permute_297: "f32[256, 256]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_95: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[256]" = torch.ops.aten.reshape.default(sum_95, [256]);  sum_95 = None
    permute_298: "f32[256, 256]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_411: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_58, [1, 512, 256]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_133: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_245, view_411);  mul_245 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_299: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_29, [0, 2, 1, 3]);  permute_default_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_412: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_299, [1, 512, 256]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_413: "f32[512, 256]" = torch.ops.aten.reshape.default(view_412, [512, 256]);  view_412 = None
    mm_60: "f32[512, 256]" = torch.ops.aten.mm.default(view_413, permute_300);  permute_300 = None
    permute_301: "f32[256, 512]" = torch.ops.aten.permute.default(view_413, [1, 0])
    mm_61: "f32[256, 256]" = torch.ops.aten.mm.default(permute_301, view_156);  permute_301 = None
    permute_302: "f32[256, 256]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_96: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_413, [0], True);  view_413 = None
    view_414: "f32[256]" = torch.ops.aten.reshape.default(sum_96, [256]);  sum_96 = None
    permute_303: "f32[256, 256]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_415: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_60, [1, 512, 256]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_134: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_133, view_415);  add_133 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_416: "f32[512, 256]" = torch.ops.aten.reshape.default(view_407, [512, 256]);  view_407 = None
    mm_62: "f32[512, 256]" = torch.ops.aten.mm.default(view_416, permute_304);  permute_304 = None
    permute_305: "f32[256, 512]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_63: "f32[256, 256]" = torch.ops.aten.mm.default(permute_305, view_156);  permute_305 = view_156 = None
    permute_306: "f32[256, 256]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.reshape.default(sum_97, [256]);  sum_97 = None
    permute_307: "f32[256, 256]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_418: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_62, [1, 512, 256]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_135: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_134, view_418);  add_134 = view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_254: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, primals_118);  primals_118 = None
    mul_255: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_254, 256)
    sum_98: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True)
    mul_256: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_254, mul_50);  mul_254 = None
    sum_99: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_50, sum_99);  sum_99 = None
    sub_81: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_255, sum_98);  mul_255 = sum_98 = None
    sub_82: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_81, mul_257);  sub_81 = mul_257 = None
    mul_258: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_42, sub_82);  div_42 = sub_82 = None
    mul_259: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_50);  mul_50 = None
    sum_100: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 1]);  mul_259 = None
    sum_101: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_16: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_260: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
    mul_261: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_258, mul_260);  mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_419: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_261, [512, 256]);  mul_261 = None
    mm_64: "f32[512, 1024]" = torch.ops.aten.mm.default(view_419, permute_308);  permute_308 = None
    permute_309: "f32[256, 512]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_65: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_309, view_154);  permute_309 = view_154 = None
    permute_310: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_102: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[256]" = torch.ops.aten.reshape.default(sum_102, [256]);  sum_102 = None
    permute_311: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_421: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_64, [1, 512, 1024]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_263: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
    mul_264: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, view_153)
    mul_265: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_264, -0.5);  mul_264 = None
    exp_20: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_265);  mul_265 = None
    mul_266: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_267: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_153, mul_266);  view_153 = mul_266 = None
    add_137: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_263, mul_267);  mul_263 = mul_267 = None
    mul_268: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_421, add_137);  view_421 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_422: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_268, [512, 1024]);  mul_268 = None
    mm_66: "f32[512, 256]" = torch.ops.aten.mm.default(view_422, permute_312);  permute_312 = None
    permute_313: "f32[1024, 512]" = torch.ops.aten.permute.default(view_422, [1, 0])
    mm_67: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_313, view_152);  permute_313 = view_152 = None
    permute_314: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True);  view_422 = None
    view_423: "f32[1024]" = torch.ops.aten.reshape.default(sum_103, [1024]);  sum_103 = None
    permute_315: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_424: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_66, [1, 512, 256]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_138: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_258, view_424);  mul_258 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_270: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, primals_112);  primals_112 = None
    mul_271: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, 256)
    sum_104: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_270, mul_45);  mul_270 = None
    sum_105: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_45, sum_105);  sum_105 = None
    sub_84: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_271, sum_104);  mul_271 = sum_104 = None
    sub_85: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_84, mul_273);  sub_84 = mul_273 = None
    mul_274: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_85);  div_43 = sub_85 = None
    mul_275: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_138, mul_45);  mul_45 = None
    sum_106: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_107: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_17: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_276: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
    mul_277: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_274, mul_276);  mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_425: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_277, [512, 256]);  mul_277 = None
    mm_68: "f32[512, 256]" = torch.ops.aten.mm.default(view_425, permute_316);  permute_316 = None
    permute_317: "f32[256, 512]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_69: "f32[256, 256]" = torch.ops.aten.mm.default(permute_317, view_150);  permute_317 = view_150 = None
    permute_318: "f32[256, 256]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
    view_426: "f32[256]" = torch.ops.aten.reshape.default(sum_108, [256]);  sum_108 = None
    permute_319: "f32[256, 256]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    view_427: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_68, [1, 512, 256]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_428: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_427, [1, 512, 4, 64]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_320: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # No stacktrace found for following nodes
    view_default_66: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_320, [4, 512, 64]);  permute_320 = None
    bmm_default_32: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_31, view_default_66);  permute_default_31 = None
    view_default_67: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_32, [1, 4, 512, 64]);  bmm_default_32 = None
    bmm_default_33: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_66, permute_default_32);  view_default_66 = permute_default_32 = None
    view_default_68: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_33, [1, 4, 512, 512]);  bmm_default_33 = None
    mul_tensor_21: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_68, mul_tensor_20);  view_default_68 = mul_tensor_20 = None
    mul_tensor_22: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_21, alias_default_11);  mul_tensor_21 = None
    sum_dim_int_list_11: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [-1], True)
    mul_tensor_23: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_11, sum_dim_int_list_11);  alias_default_11 = sum_dim_int_list_11 = None
    sub_tensor_11: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_22, mul_tensor_23);  mul_tensor_22 = mul_tensor_23 = None
    view_default_69: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_11, [4, 512, 512]);  sub_tensor_11 = None
    bmm_default_34: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_33, view_default_69);  permute_default_33 = None
    view_default_70: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_34, [1, 4, 64, 512]);  bmm_default_34 = None
    mul_scalar_22: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_70, 0.3535533905932738);  view_default_70 = None
    permute_default_35: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_22, [0, 1, 3, 2]);  mul_scalar_22 = None
    bmm_default_35: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_69, permute_default_34);  view_default_69 = permute_default_34 = None
    view_default_71: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_35, [1, 4, 512, 64]);  bmm_default_35 = None
    mul_scalar_23: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_71, 0.3535533905932738);  view_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_326: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_23, [0, 2, 1, 3]);  mul_scalar_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_40: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_326, memory_format = torch.contiguous_format);  permute_326 = None
    view_435: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_40, [1, 512, 256]);  clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_327: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_67, [0, 2, 1, 3]);  view_default_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_41: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_436: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_41, [1, 512, 256]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_437: "f32[512, 256]" = torch.ops.aten.reshape.default(view_436, [512, 256]);  view_436 = None
    mm_70: "f32[512, 256]" = torch.ops.aten.mm.default(view_437, permute_328);  permute_328 = None
    permute_329: "f32[256, 512]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_71: "f32[256, 256]" = torch.ops.aten.mm.default(permute_329, view_134);  permute_329 = None
    permute_330: "f32[256, 256]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[256]" = torch.ops.aten.reshape.default(sum_110, [256]);  sum_110 = None
    permute_331: "f32[256, 256]" = torch.ops.aten.permute.default(permute_330, [1, 0]);  permute_330 = None
    view_439: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_70, [1, 512, 256]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_139: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_274, view_439);  mul_274 = view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_332: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_35, [0, 2, 1, 3]);  permute_default_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_440: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_332, [1, 512, 256]);  permute_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_441: "f32[512, 256]" = torch.ops.aten.reshape.default(view_440, [512, 256]);  view_440 = None
    mm_72: "f32[512, 256]" = torch.ops.aten.mm.default(view_441, permute_333);  permute_333 = None
    permute_334: "f32[256, 512]" = torch.ops.aten.permute.default(view_441, [1, 0])
    mm_73: "f32[256, 256]" = torch.ops.aten.mm.default(permute_334, view_134);  permute_334 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_441, [0], True);  view_441 = None
    view_442: "f32[256]" = torch.ops.aten.reshape.default(sum_111, [256]);  sum_111 = None
    permute_336: "f32[256, 256]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
    view_443: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_72, [1, 512, 256]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_140: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_139, view_443);  add_139 = view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_444: "f32[512, 256]" = torch.ops.aten.reshape.default(view_435, [512, 256]);  view_435 = None
    mm_74: "f32[512, 256]" = torch.ops.aten.mm.default(view_444, permute_337);  permute_337 = None
    permute_338: "f32[256, 512]" = torch.ops.aten.permute.default(view_444, [1, 0])
    mm_75: "f32[256, 256]" = torch.ops.aten.mm.default(permute_338, view_134);  permute_338 = view_134 = None
    permute_339: "f32[256, 256]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_112: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True);  view_444 = None
    view_445: "f32[256]" = torch.ops.aten.reshape.default(sum_112, [256]);  sum_112 = None
    permute_340: "f32[256, 256]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_446: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_74, [1, 512, 256]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_141: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_140, view_446);  add_140 = view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_283: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, primals_102);  primals_102 = None
    mul_284: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_283, 256)
    sum_113: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_285: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_283, mul_43);  mul_283 = None
    sum_114: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_43, sum_114);  sum_114 = None
    sub_88: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_284, sum_113);  mul_284 = sum_113 = None
    sub_89: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_88, mul_286);  sub_88 = mul_286 = None
    mul_287: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_89);  div_45 = sub_89 = None
    mul_288: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_141, mul_43);  mul_43 = None
    sum_115: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_288, [0, 1]);  mul_288 = None
    sum_116: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_141, [0, 1]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_19: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_289: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
    mul_290: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_287, mul_289);  mul_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_447: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_290, [512, 256]);  mul_290 = None
    mm_76: "f32[512, 1024]" = torch.ops.aten.mm.default(view_447, permute_341);  permute_341 = None
    permute_342: "f32[256, 512]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_77: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_342, view_132);  permute_342 = view_132 = None
    permute_343: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[256]" = torch.ops.aten.reshape.default(sum_117, [256]);  sum_117 = None
    permute_344: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    view_449: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_76, [1, 512, 1024]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_292: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
    mul_293: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, view_131)
    mul_294: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_293, -0.5);  mul_293 = None
    exp_21: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_294);  mul_294 = None
    mul_295: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_296: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_131, mul_295);  view_131 = mul_295 = None
    add_143: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_292, mul_296);  mul_292 = mul_296 = None
    mul_297: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_449, add_143);  view_449 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_450: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_297, [512, 1024]);  mul_297 = None
    mm_78: "f32[512, 256]" = torch.ops.aten.mm.default(view_450, permute_345);  permute_345 = None
    permute_346: "f32[1024, 512]" = torch.ops.aten.permute.default(view_450, [1, 0])
    mm_79: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_346, view_130);  permute_346 = view_130 = None
    permute_347: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_118: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_450, [0], True);  view_450 = None
    view_451: "f32[1024]" = torch.ops.aten.reshape.default(sum_118, [1024]);  sum_118 = None
    permute_348: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
    view_452: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_78, [1, 512, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_144: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_287, view_452);  mul_287 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_299: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, primals_96);  primals_96 = None
    mul_300: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, 256)
    sum_119: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True)
    mul_301: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_299, mul_38);  mul_299 = None
    sum_120: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True);  mul_301 = None
    mul_302: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_38, sum_120);  sum_120 = None
    sub_91: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_300, sum_119);  mul_300 = sum_119 = None
    sub_92: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_91, mul_302);  sub_91 = mul_302 = None
    mul_303: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_92);  div_46 = sub_92 = None
    mul_304: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_144, mul_38);  mul_38 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1]);  mul_304 = None
    sum_122: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_20: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_305: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
    mul_306: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_303, mul_305);  mul_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_453: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_306, [512, 256]);  mul_306 = None
    mm_80: "f32[512, 256]" = torch.ops.aten.mm.default(view_453, permute_349);  permute_349 = None
    permute_350: "f32[256, 512]" = torch.ops.aten.permute.default(view_453, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_350, view_128);  permute_350 = view_128 = None
    permute_351: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_123: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
    view_454: "f32[256]" = torch.ops.aten.reshape.default(sum_123, [256]);  sum_123 = None
    permute_352: "f32[256, 256]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    view_455: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_80, [1, 512, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_456: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_455, [1, 512, 4, 64]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_353: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    
    # No stacktrace found for following nodes
    view_default_78: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_353, [4, 512, 64]);  permute_353 = None
    bmm_default_38: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_37, view_default_78);  permute_default_37 = None
    view_default_79: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_38, [1, 4, 512, 64]);  bmm_default_38 = None
    bmm_default_39: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_78, permute_default_38);  view_default_78 = permute_default_38 = None
    view_default_80: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_39, [1, 4, 512, 512]);  bmm_default_39 = None
    mul_tensor_25: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_80, mul_tensor_24);  view_default_80 = mul_tensor_24 = None
    mul_tensor_26: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_25, alias_default_13);  mul_tensor_25 = None
    sum_dim_int_list_13: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [-1], True)
    mul_tensor_27: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_13, sum_dim_int_list_13);  alias_default_13 = sum_dim_int_list_13 = None
    sub_tensor_13: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_26, mul_tensor_27);  mul_tensor_26 = mul_tensor_27 = None
    view_default_81: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_13, [4, 512, 512]);  sub_tensor_13 = None
    bmm_default_40: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_39, view_default_81);  permute_default_39 = None
    view_default_82: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_40, [1, 4, 64, 512]);  bmm_default_40 = None
    mul_scalar_26: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_82, 0.3535533905932738);  view_default_82 = None
    permute_default_41: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_26, [0, 1, 3, 2]);  mul_scalar_26 = None
    bmm_default_41: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_81, permute_default_40);  view_default_81 = permute_default_40 = None
    view_default_83: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_41, [1, 4, 512, 64]);  bmm_default_41 = None
    mul_scalar_27: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_83, 0.3535533905932738);  view_default_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_359: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_27, [0, 2, 1, 3]);  mul_scalar_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_45: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_463: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_45, [1, 512, 256]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_360: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_79, [0, 2, 1, 3]);  view_default_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_46: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None
    view_464: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_46, [1, 512, 256]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_465: "f32[512, 256]" = torch.ops.aten.reshape.default(view_464, [512, 256]);  view_464 = None
    mm_82: "f32[512, 256]" = torch.ops.aten.mm.default(view_465, permute_361);  permute_361 = None
    permute_362: "f32[256, 512]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_362, view_112);  permute_362 = None
    permute_363: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[256]" = torch.ops.aten.reshape.default(sum_125, [256]);  sum_125 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
    view_467: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_82, [1, 512, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_145: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_303, view_467);  mul_303 = view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_365: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_41, [0, 2, 1, 3]);  permute_default_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_468: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_365, [1, 512, 256]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_469: "f32[512, 256]" = torch.ops.aten.reshape.default(view_468, [512, 256]);  view_468 = None
    mm_84: "f32[512, 256]" = torch.ops.aten.mm.default(view_469, permute_366);  permute_366 = None
    permute_367: "f32[256, 512]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_85: "f32[256, 256]" = torch.ops.aten.mm.default(permute_367, view_112);  permute_367 = None
    permute_368: "f32[256, 256]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_126: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[256]" = torch.ops.aten.reshape.default(sum_126, [256]);  sum_126 = None
    permute_369: "f32[256, 256]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_471: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_84, [1, 512, 256]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_146: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_145, view_471);  add_145 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_472: "f32[512, 256]" = torch.ops.aten.reshape.default(view_463, [512, 256]);  view_463 = None
    mm_86: "f32[512, 256]" = torch.ops.aten.mm.default(view_472, permute_370);  permute_370 = None
    permute_371: "f32[256, 512]" = torch.ops.aten.permute.default(view_472, [1, 0])
    mm_87: "f32[256, 256]" = torch.ops.aten.mm.default(permute_371, view_112);  permute_371 = view_112 = None
    permute_372: "f32[256, 256]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_127: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
    view_473: "f32[256]" = torch.ops.aten.reshape.default(sum_127, [256]);  sum_127 = None
    permute_373: "f32[256, 256]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_474: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_86, [1, 512, 256]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_147: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_146, view_474);  add_146 = view_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_312: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, primals_86);  primals_86 = None
    mul_313: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, 256)
    sum_128: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_312, mul_36);  mul_312 = None
    sum_129: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_36, sum_129);  sum_129 = None
    sub_95: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_313, sum_128);  mul_313 = sum_128 = None
    sub_96: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_95, mul_315);  sub_95 = mul_315 = None
    mul_316: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_96);  div_48 = sub_96 = None
    mul_317: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_147, mul_36);  mul_36 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_131: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_22: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_318: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
    mul_319: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_316, mul_318);  mul_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_475: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_319, [512, 256]);  mul_319 = None
    mm_88: "f32[512, 1024]" = torch.ops.aten.mm.default(view_475, permute_374);  permute_374 = None
    permute_375: "f32[256, 512]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_89: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_375, view_110);  permute_375 = view_110 = None
    permute_376: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_132: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[256]" = torch.ops.aten.reshape.default(sum_132, [256]);  sum_132 = None
    permute_377: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_477: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_88, [1, 512, 1024]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_321: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_322: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, view_109)
    mul_323: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_322, -0.5);  mul_322 = None
    exp_22: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_323);  mul_323 = None
    mul_324: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_325: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_324);  view_109 = mul_324 = None
    add_149: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_321, mul_325);  mul_321 = mul_325 = None
    mul_326: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_477, add_149);  view_477 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_478: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_326, [512, 1024]);  mul_326 = None
    mm_90: "f32[512, 256]" = torch.ops.aten.mm.default(view_478, permute_378);  permute_378 = None
    permute_379: "f32[1024, 512]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_91: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_379, view_108);  permute_379 = view_108 = None
    permute_380: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_133: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[1024]" = torch.ops.aten.reshape.default(sum_133, [1024]);  sum_133 = None
    permute_381: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
    view_480: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_90, [1, 512, 256]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_150: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_316, view_480);  mul_316 = view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_328: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, primals_80);  primals_80 = None
    mul_329: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, 256)
    sum_134: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True)
    mul_330: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_328, mul_31);  mul_328 = None
    sum_135: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2], True);  mul_330 = None
    mul_331: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_31, sum_135);  sum_135 = None
    sub_98: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_329, sum_134);  mul_329 = sum_134 = None
    sub_99: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_98, mul_331);  sub_98 = mul_331 = None
    mul_332: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_49, sub_99);  div_49 = sub_99 = None
    mul_333: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_150, mul_31);  mul_31 = None
    sum_136: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 1]);  mul_333 = None
    sum_137: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_23: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_334: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
    mul_335: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_332, mul_334);  mul_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_481: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_335, [512, 256]);  mul_335 = None
    mm_92: "f32[512, 256]" = torch.ops.aten.mm.default(view_481, permute_382);  permute_382 = None
    permute_383: "f32[256, 512]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_93: "f32[256, 256]" = torch.ops.aten.mm.default(permute_383, view_106);  permute_383 = view_106 = None
    permute_384: "f32[256, 256]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_138: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[256]" = torch.ops.aten.reshape.default(sum_138, [256]);  sum_138 = None
    permute_385: "f32[256, 256]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    view_483: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_92, [1, 512, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_484: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_483, [1, 512, 4, 64]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_386: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # No stacktrace found for following nodes
    view_default_90: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_386, [4, 512, 64]);  permute_386 = None
    bmm_default_44: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_43, view_default_90);  permute_default_43 = None
    view_default_91: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_44, [1, 4, 512, 64]);  bmm_default_44 = None
    bmm_default_45: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_90, permute_default_44);  view_default_90 = permute_default_44 = None
    view_default_92: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_45, [1, 4, 512, 512]);  bmm_default_45 = None
    mul_tensor_29: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_92, mul_tensor_28);  view_default_92 = mul_tensor_28 = None
    mul_tensor_30: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_29, alias_default_15);  mul_tensor_29 = None
    sum_dim_int_list_15: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [-1], True)
    mul_tensor_31: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_15, sum_dim_int_list_15);  alias_default_15 = sum_dim_int_list_15 = None
    sub_tensor_15: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
    view_default_93: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_15, [4, 512, 512]);  sub_tensor_15 = None
    bmm_default_46: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_45, view_default_93);  permute_default_45 = None
    view_default_94: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_46, [1, 4, 64, 512]);  bmm_default_46 = None
    mul_scalar_30: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_94, 0.3535533905932738);  view_default_94 = None
    permute_default_47: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_30, [0, 1, 3, 2]);  mul_scalar_30 = None
    bmm_default_47: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_93, permute_default_46);  view_default_93 = permute_default_46 = None
    view_default_95: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_47, [1, 4, 512, 64]);  bmm_default_47 = None
    mul_scalar_31: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_95, 0.3535533905932738);  view_default_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_392: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_31, [0, 2, 1, 3]);  mul_scalar_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_50: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_392, memory_format = torch.contiguous_format);  permute_392 = None
    view_491: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_50, [1, 512, 256]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_393: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_91, [0, 2, 1, 3]);  view_default_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_51: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_393, memory_format = torch.contiguous_format);  permute_393 = None
    view_492: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_51, [1, 512, 256]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_493: "f32[512, 256]" = torch.ops.aten.reshape.default(view_492, [512, 256]);  view_492 = None
    mm_94: "f32[512, 256]" = torch.ops.aten.mm.default(view_493, permute_394);  permute_394 = None
    permute_395: "f32[256, 512]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_395, view_90);  permute_395 = None
    permute_396: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[256]" = torch.ops.aten.reshape.default(sum_140, [256]);  sum_140 = None
    permute_397: "f32[256, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_495: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_94, [1, 512, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_151: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_332, view_495);  mul_332 = view_495 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_398: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_47, [0, 2, 1, 3]);  permute_default_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_496: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_398, [1, 512, 256]);  permute_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_497: "f32[512, 256]" = torch.ops.aten.reshape.default(view_496, [512, 256]);  view_496 = None
    mm_96: "f32[512, 256]" = torch.ops.aten.mm.default(view_497, permute_399);  permute_399 = None
    permute_400: "f32[256, 512]" = torch.ops.aten.permute.default(view_497, [1, 0])
    mm_97: "f32[256, 256]" = torch.ops.aten.mm.default(permute_400, view_90);  permute_400 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
    view_498: "f32[256]" = torch.ops.aten.reshape.default(sum_141, [256]);  sum_141 = None
    permute_402: "f32[256, 256]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
    view_499: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_96, [1, 512, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_152: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_151, view_499);  add_151 = view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_500: "f32[512, 256]" = torch.ops.aten.reshape.default(view_491, [512, 256]);  view_491 = None
    mm_98: "f32[512, 256]" = torch.ops.aten.mm.default(view_500, permute_403);  permute_403 = None
    permute_404: "f32[256, 512]" = torch.ops.aten.permute.default(view_500, [1, 0])
    mm_99: "f32[256, 256]" = torch.ops.aten.mm.default(permute_404, view_90);  permute_404 = view_90 = None
    permute_405: "f32[256, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_142: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
    view_501: "f32[256]" = torch.ops.aten.reshape.default(sum_142, [256]);  sum_142 = None
    permute_406: "f32[256, 256]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
    view_502: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_98, [1, 512, 256]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_153: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_152, view_502);  add_152 = view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_341: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, primals_70);  primals_70 = None
    mul_342: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_341, 256)
    sum_143: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True)
    mul_343: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_341, mul_29);  mul_341 = None
    sum_144: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True);  mul_343 = None
    mul_344: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_29, sum_144);  sum_144 = None
    sub_102: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_342, sum_143);  mul_342 = sum_143 = None
    sub_103: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_102, mul_344);  sub_102 = mul_344 = None
    mul_345: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_51, sub_103);  div_51 = sub_103 = None
    mul_346: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_153, mul_29);  mul_29 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 1]);  mul_346 = None
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_25: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_347: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
    mul_348: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_345, mul_347);  mul_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_503: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_348, [512, 256]);  mul_348 = None
    mm_100: "f32[512, 1024]" = torch.ops.aten.mm.default(view_503, permute_407);  permute_407 = None
    permute_408: "f32[256, 512]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_101: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_408, view_88);  permute_408 = view_88 = None
    permute_409: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_147: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[256]" = torch.ops.aten.reshape.default(sum_147, [256]);  sum_147 = None
    permute_410: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_409, [1, 0]);  permute_409 = None
    view_505: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_100, [1, 512, 1024]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_350: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
    mul_351: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, view_87)
    mul_352: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_351, -0.5);  mul_351 = None
    exp_23: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_352);  mul_352 = None
    mul_353: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_354: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_87, mul_353);  view_87 = mul_353 = None
    add_155: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_350, mul_354);  mul_350 = mul_354 = None
    mul_355: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_505, add_155);  view_505 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_506: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_355, [512, 1024]);  mul_355 = None
    mm_102: "f32[512, 256]" = torch.ops.aten.mm.default(view_506, permute_411);  permute_411 = None
    permute_412: "f32[1024, 512]" = torch.ops.aten.permute.default(view_506, [1, 0])
    mm_103: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_412, view_86);  permute_412 = view_86 = None
    permute_413: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_148: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
    view_507: "f32[1024]" = torch.ops.aten.reshape.default(sum_148, [1024]);  sum_148 = None
    permute_414: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
    view_508: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_102, [1, 512, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_345, view_508);  mul_345 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_357: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, primals_64);  primals_64 = None
    mul_358: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, 256)
    sum_149: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
    mul_359: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_357, mul_24);  mul_357 = None
    sum_150: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
    mul_360: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_24, sum_150);  sum_150 = None
    sub_105: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_358, sum_149);  mul_358 = sum_149 = None
    sub_106: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_105, mul_360);  sub_105 = mul_360 = None
    mul_361: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_52, sub_106);  div_52 = sub_106 = None
    mul_362: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_156, mul_24);  mul_24 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
    sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_26: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_363: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
    mul_364: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_361, mul_363);  mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_509: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_364, [512, 256]);  mul_364 = None
    mm_104: "f32[512, 256]" = torch.ops.aten.mm.default(view_509, permute_415);  permute_415 = None
    permute_416: "f32[256, 512]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_105: "f32[256, 256]" = torch.ops.aten.mm.default(permute_416, view_84);  permute_416 = view_84 = None
    permute_417: "f32[256, 256]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_153: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[256]" = torch.ops.aten.reshape.default(sum_153, [256]);  sum_153 = None
    permute_418: "f32[256, 256]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_511: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_104, [1, 512, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_512: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_511, [1, 512, 4, 64]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_419: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
    
    # No stacktrace found for following nodes
    view_default_102: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_419, [4, 512, 64]);  permute_419 = None
    bmm_default_50: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_49, view_default_102);  permute_default_49 = None
    view_default_103: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_50, [1, 4, 512, 64]);  bmm_default_50 = None
    bmm_default_51: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_102, permute_default_50);  view_default_102 = permute_default_50 = None
    view_default_104: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_51, [1, 4, 512, 512]);  bmm_default_51 = None
    mul_tensor_33: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_104, mul_tensor_32);  view_default_104 = mul_tensor_32 = None
    mul_tensor_34: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_33, alias_default_17);  mul_tensor_33 = None
    sum_dim_int_list_17: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [-1], True)
    mul_tensor_35: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_17, sum_dim_int_list_17);  alias_default_17 = sum_dim_int_list_17 = None
    sub_tensor_17: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_34, mul_tensor_35);  mul_tensor_34 = mul_tensor_35 = None
    view_default_105: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_17, [4, 512, 512]);  sub_tensor_17 = None
    bmm_default_52: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_51, view_default_105);  permute_default_51 = None
    view_default_106: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_52, [1, 4, 64, 512]);  bmm_default_52 = None
    mul_scalar_34: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_106, 0.3535533905932738);  view_default_106 = None
    permute_default_53: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_34, [0, 1, 3, 2]);  mul_scalar_34 = None
    bmm_default_53: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_105, permute_default_52);  view_default_105 = permute_default_52 = None
    view_default_107: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_53, [1, 4, 512, 64]);  bmm_default_53 = None
    mul_scalar_35: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_107, 0.3535533905932738);  view_default_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_425: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_35, [0, 2, 1, 3]);  mul_scalar_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_55: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_425, memory_format = torch.contiguous_format);  permute_425 = None
    view_519: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_55, [1, 512, 256]);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_426: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_103, [0, 2, 1, 3]);  view_default_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_56: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_426, memory_format = torch.contiguous_format);  permute_426 = None
    view_520: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_56, [1, 512, 256]);  clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_521: "f32[512, 256]" = torch.ops.aten.reshape.default(view_520, [512, 256]);  view_520 = None
    mm_106: "f32[512, 256]" = torch.ops.aten.mm.default(view_521, permute_427);  permute_427 = None
    permute_428: "f32[256, 512]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_107: "f32[256, 256]" = torch.ops.aten.mm.default(permute_428, view_68);  permute_428 = None
    permute_429: "f32[256, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_155: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[256]" = torch.ops.aten.reshape.default(sum_155, [256]);  sum_155 = None
    permute_430: "f32[256, 256]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    view_523: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_106, [1, 512, 256]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_157: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_361, view_523);  mul_361 = view_523 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_431: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_53, [0, 2, 1, 3]);  permute_default_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_524: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_431, [1, 512, 256]);  permute_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_525: "f32[512, 256]" = torch.ops.aten.reshape.default(view_524, [512, 256]);  view_524 = None
    mm_108: "f32[512, 256]" = torch.ops.aten.mm.default(view_525, permute_432);  permute_432 = None
    permute_433: "f32[256, 512]" = torch.ops.aten.permute.default(view_525, [1, 0])
    mm_109: "f32[256, 256]" = torch.ops.aten.mm.default(permute_433, view_68);  permute_433 = None
    permute_434: "f32[256, 256]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_156: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
    view_526: "f32[256]" = torch.ops.aten.reshape.default(sum_156, [256]);  sum_156 = None
    permute_435: "f32[256, 256]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_527: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_108, [1, 512, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_158: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_157, view_527);  add_157 = view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_528: "f32[512, 256]" = torch.ops.aten.reshape.default(view_519, [512, 256]);  view_519 = None
    mm_110: "f32[512, 256]" = torch.ops.aten.mm.default(view_528, permute_436);  permute_436 = None
    permute_437: "f32[256, 512]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_111: "f32[256, 256]" = torch.ops.aten.mm.default(permute_437, view_68);  permute_437 = view_68 = None
    permute_438: "f32[256, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_157: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[256]" = torch.ops.aten.reshape.default(sum_157, [256]);  sum_157 = None
    permute_439: "f32[256, 256]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_530: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_110, [1, 512, 256]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_159: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_158, view_530);  add_158 = view_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_370: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, primals_54);  primals_54 = None
    mul_371: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_370, 256)
    sum_158: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
    mul_372: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_370, mul_22);  mul_370 = None
    sum_159: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
    mul_373: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_22, sum_159);  sum_159 = None
    sub_109: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_371, sum_158);  mul_371 = sum_158 = None
    sub_110: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_109, mul_373);  sub_109 = mul_373 = None
    mul_374: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_54, sub_110);  div_54 = sub_110 = None
    mul_375: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_159, mul_22);  mul_22 = None
    sum_160: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
    sum_161: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_28: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_376: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
    mul_377: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_374, mul_376);  mul_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_531: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_377, [512, 256]);  mul_377 = None
    mm_112: "f32[512, 1024]" = torch.ops.aten.mm.default(view_531, permute_440);  permute_440 = None
    permute_441: "f32[256, 512]" = torch.ops.aten.permute.default(view_531, [1, 0])
    mm_113: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_441, view_66);  permute_441 = view_66 = None
    permute_442: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_162: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_531, [0], True);  view_531 = None
    view_532: "f32[256]" = torch.ops.aten.reshape.default(sum_162, [256]);  sum_162 = None
    permute_443: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_533: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_112, [1, 512, 1024]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_379: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
    mul_380: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, view_65)
    mul_381: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_380, -0.5);  mul_380 = None
    exp_24: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_381);  mul_381 = None
    mul_382: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_383: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_65, mul_382);  view_65 = mul_382 = None
    add_161: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_379, mul_383);  mul_379 = mul_383 = None
    mul_384: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_533, add_161);  view_533 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_534: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_384, [512, 1024]);  mul_384 = None
    mm_114: "f32[512, 256]" = torch.ops.aten.mm.default(view_534, permute_444);  permute_444 = None
    permute_445: "f32[1024, 512]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_115: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_445, view_64);  permute_445 = view_64 = None
    permute_446: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_163: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[1024]" = torch.ops.aten.reshape.default(sum_163, [1024]);  sum_163 = None
    permute_447: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_446, [1, 0]);  permute_446 = None
    view_536: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_114, [1, 512, 256]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_162: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_374, view_536);  mul_374 = view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_386: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, primals_48);  primals_48 = None
    mul_387: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, 256)
    sum_164: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_386, [2], True)
    mul_388: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_386, mul_17);  mul_386 = None
    sum_165: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True);  mul_388 = None
    mul_389: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_17, sum_165);  sum_165 = None
    sub_112: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_387, sum_164);  mul_387 = sum_164 = None
    sub_113: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_112, mul_389);  sub_112 = mul_389 = None
    mul_390: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_55, sub_113);  div_55 = sub_113 = None
    mul_391: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_162, mul_17);  mul_17 = None
    sum_166: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 1]);  mul_391 = None
    sum_167: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_29: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_392: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
    mul_393: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_390, mul_392);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_537: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_393, [512, 256]);  mul_393 = None
    mm_116: "f32[512, 256]" = torch.ops.aten.mm.default(view_537, permute_448);  permute_448 = None
    permute_449: "f32[256, 512]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_117: "f32[256, 256]" = torch.ops.aten.mm.default(permute_449, view_62);  permute_449 = view_62 = None
    permute_450: "f32[256, 256]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_168: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[256]" = torch.ops.aten.reshape.default(sum_168, [256]);  sum_168 = None
    permute_451: "f32[256, 256]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_539: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_116, [1, 512, 256]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_540: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_539, [1, 512, 4, 64]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_452: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # No stacktrace found for following nodes
    view_default_114: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_452, [4, 512, 64]);  permute_452 = None
    bmm_default_56: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_55, view_default_114);  permute_default_55 = None
    view_default_115: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_56, [1, 4, 512, 64]);  bmm_default_56 = None
    bmm_default_57: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_114, permute_default_56);  view_default_114 = permute_default_56 = None
    view_default_116: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_57, [1, 4, 512, 512]);  bmm_default_57 = None
    mul_tensor_37: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_116, mul_tensor_36);  view_default_116 = mul_tensor_36 = None
    mul_tensor_38: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_37, alias_default_19);  mul_tensor_37 = None
    sum_dim_int_list_19: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [-1], True)
    mul_tensor_39: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_19, sum_dim_int_list_19);  alias_default_19 = sum_dim_int_list_19 = None
    sub_tensor_19: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None
    view_default_117: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_19, [4, 512, 512]);  sub_tensor_19 = None
    bmm_default_58: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_57, view_default_117);  permute_default_57 = None
    view_default_118: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_58, [1, 4, 64, 512]);  bmm_default_58 = None
    mul_scalar_38: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_118, 0.3535533905932738);  view_default_118 = None
    permute_default_59: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_38, [0, 1, 3, 2]);  mul_scalar_38 = None
    bmm_default_59: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_117, permute_default_58);  view_default_117 = permute_default_58 = None
    view_default_119: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_59, [1, 4, 512, 64]);  bmm_default_59 = None
    mul_scalar_39: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_119, 0.3535533905932738);  view_default_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_458: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_39, [0, 2, 1, 3]);  mul_scalar_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_60: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_458, memory_format = torch.contiguous_format);  permute_458 = None
    view_547: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_60, [1, 512, 256]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_459: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_115, [0, 2, 1, 3]);  view_default_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_61: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_548: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_61, [1, 512, 256]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_549: "f32[512, 256]" = torch.ops.aten.reshape.default(view_548, [512, 256]);  view_548 = None
    mm_118: "f32[512, 256]" = torch.ops.aten.mm.default(view_549, permute_460);  permute_460 = None
    permute_461: "f32[256, 512]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_119: "f32[256, 256]" = torch.ops.aten.mm.default(permute_461, view_46);  permute_461 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_170: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[256]" = torch.ops.aten.reshape.default(sum_170, [256]);  sum_170 = None
    permute_463: "f32[256, 256]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
    view_551: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_118, [1, 512, 256]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_163: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_390, view_551);  mul_390 = view_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_464: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_59, [0, 2, 1, 3]);  permute_default_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_552: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_464, [1, 512, 256]);  permute_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_553: "f32[512, 256]" = torch.ops.aten.reshape.default(view_552, [512, 256]);  view_552 = None
    mm_120: "f32[512, 256]" = torch.ops.aten.mm.default(view_553, permute_465);  permute_465 = None
    permute_466: "f32[256, 512]" = torch.ops.aten.permute.default(view_553, [1, 0])
    mm_121: "f32[256, 256]" = torch.ops.aten.mm.default(permute_466, view_46);  permute_466 = None
    permute_467: "f32[256, 256]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
    view_554: "f32[256]" = torch.ops.aten.reshape.default(sum_171, [256]);  sum_171 = None
    permute_468: "f32[256, 256]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
    view_555: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_120, [1, 512, 256]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_164: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_163, view_555);  add_163 = view_555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_556: "f32[512, 256]" = torch.ops.aten.reshape.default(view_547, [512, 256]);  view_547 = None
    mm_122: "f32[512, 256]" = torch.ops.aten.mm.default(view_556, permute_469);  permute_469 = None
    permute_470: "f32[256, 512]" = torch.ops.aten.permute.default(view_556, [1, 0])
    mm_123: "f32[256, 256]" = torch.ops.aten.mm.default(permute_470, view_46);  permute_470 = view_46 = None
    permute_471: "f32[256, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_172: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
    view_557: "f32[256]" = torch.ops.aten.reshape.default(sum_172, [256]);  sum_172 = None
    permute_472: "f32[256, 256]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
    view_558: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_122, [1, 512, 256]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_165: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_164, view_558);  add_164 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_399: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, primals_38);  primals_38 = None
    mul_400: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_399, 256)
    sum_173: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True)
    mul_401: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_399, mul_15);  mul_399 = None
    sum_174: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [2], True);  mul_401 = None
    mul_402: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_15, sum_174);  sum_174 = None
    sub_116: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_400, sum_173);  mul_400 = sum_173 = None
    sub_117: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_116, mul_402);  sub_116 = mul_402 = None
    mul_403: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_57, sub_117);  div_57 = sub_117 = None
    mul_404: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_165, mul_15);  mul_15 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1]);  mul_404 = None
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_165, [0, 1]);  add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_31: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_405: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
    mul_406: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_403, mul_405);  mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_559: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_406, [512, 256]);  mul_406 = None
    mm_124: "f32[512, 1024]" = torch.ops.aten.mm.default(view_559, permute_473);  permute_473 = None
    permute_474: "f32[256, 512]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_125: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_474, view_44);  permute_474 = view_44 = None
    permute_475: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_177: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[256]" = torch.ops.aten.reshape.default(sum_177, [256]);  sum_177 = None
    permute_476: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
    view_561: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_124, [1, 512, 1024]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_408: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
    mul_409: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_410: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_409, -0.5);  mul_409 = None
    exp_25: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_410);  mul_410 = None
    mul_411: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_412: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_43, mul_411);  view_43 = mul_411 = None
    add_167: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_408, mul_412);  mul_408 = mul_412 = None
    mul_413: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_561, add_167);  view_561 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_562: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_413, [512, 1024]);  mul_413 = None
    mm_126: "f32[512, 256]" = torch.ops.aten.mm.default(view_562, permute_477);  permute_477 = None
    permute_478: "f32[1024, 512]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_127: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_478, view_42);  permute_478 = view_42 = None
    permute_479: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_178: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[1024]" = torch.ops.aten.reshape.default(sum_178, [1024]);  sum_178 = None
    permute_480: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
    view_564: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_126, [1, 512, 256]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_168: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_403, view_564);  mul_403 = view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_415: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, primals_32);  primals_32 = None
    mul_416: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, 256)
    sum_179: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [2], True)
    mul_417: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_415, mul_10);  mul_415 = None
    sum_180: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True);  mul_417 = None
    mul_418: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_10, sum_180);  sum_180 = None
    sub_119: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_416, sum_179);  mul_416 = sum_179 = None
    sub_120: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_119, mul_418);  sub_119 = mul_418 = None
    mul_419: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_58, sub_120);  div_58 = sub_120 = None
    mul_420: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_168, mul_10);  mul_10 = None
    sum_181: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 1]);  mul_420 = None
    sum_182: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_32: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_421: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
    mul_422: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_419, mul_421);  mul_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_565: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_422, [512, 256]);  mul_422 = None
    mm_128: "f32[512, 256]" = torch.ops.aten.mm.default(view_565, permute_481);  permute_481 = None
    permute_482: "f32[256, 512]" = torch.ops.aten.permute.default(view_565, [1, 0])
    mm_129: "f32[256, 256]" = torch.ops.aten.mm.default(permute_482, view_40);  permute_482 = view_40 = None
    permute_483: "f32[256, 256]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_183: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
    view_566: "f32[256]" = torch.ops.aten.reshape.default(sum_183, [256]);  sum_183 = None
    permute_484: "f32[256, 256]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_567: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_128, [1, 512, 256]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_568: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_567, [1, 512, 4, 64]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_485: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    
    # No stacktrace found for following nodes
    view_default_126: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_485, [4, 512, 64]);  permute_485 = None
    bmm_default_62: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_61, view_default_126);  permute_default_61 = None
    view_default_127: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_62, [1, 4, 512, 64]);  bmm_default_62 = None
    bmm_default_63: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_126, permute_default_62);  view_default_126 = permute_default_62 = None
    view_default_128: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_63, [1, 4, 512, 512]);  bmm_default_63 = None
    mul_tensor_41: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_128, mul_tensor_40);  view_default_128 = mul_tensor_40 = None
    mul_tensor_42: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_41, alias_default_21);  mul_tensor_41 = None
    sum_dim_int_list_21: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [-1], True)
    mul_tensor_43: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_21, sum_dim_int_list_21);  alias_default_21 = sum_dim_int_list_21 = None
    sub_tensor_21: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_42, mul_tensor_43);  mul_tensor_42 = mul_tensor_43 = None
    view_default_129: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_21, [4, 512, 512]);  sub_tensor_21 = None
    bmm_default_64: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_63, view_default_129);  permute_default_63 = None
    view_default_130: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_64, [1, 4, 64, 512]);  bmm_default_64 = None
    mul_scalar_42: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_130, 0.3535533905932738);  view_default_130 = None
    permute_default_65: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_42, [0, 1, 3, 2]);  mul_scalar_42 = None
    bmm_default_65: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_129, permute_default_64);  view_default_129 = permute_default_64 = None
    view_default_131: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_65, [1, 4, 512, 64]);  bmm_default_65 = None
    mul_scalar_43: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_131, 0.3535533905932738);  view_default_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_491: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_43, [0, 2, 1, 3]);  mul_scalar_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_65: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_575: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_65, [1, 512, 256]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_492: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_127, [0, 2, 1, 3]);  view_default_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_66: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_492, memory_format = torch.contiguous_format);  permute_492 = None
    view_576: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_66, [1, 512, 256]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_577: "f32[512, 256]" = torch.ops.aten.reshape.default(view_576, [512, 256]);  view_576 = None
    mm_130: "f32[512, 256]" = torch.ops.aten.mm.default(view_577, permute_493);  permute_493 = None
    permute_494: "f32[256, 512]" = torch.ops.aten.permute.default(view_577, [1, 0])
    mm_131: "f32[256, 256]" = torch.ops.aten.mm.default(permute_494, view_24);  permute_494 = None
    permute_495: "f32[256, 256]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_185: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
    view_578: "f32[256]" = torch.ops.aten.reshape.default(sum_185, [256]);  sum_185 = None
    permute_496: "f32[256, 256]" = torch.ops.aten.permute.default(permute_495, [1, 0]);  permute_495 = None
    view_579: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_130, [1, 512, 256]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_169: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_419, view_579);  mul_419 = view_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_497: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_65, [0, 2, 1, 3]);  permute_default_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_580: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_497, [1, 512, 256]);  permute_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_581: "f32[512, 256]" = torch.ops.aten.reshape.default(view_580, [512, 256]);  view_580 = None
    mm_132: "f32[512, 256]" = torch.ops.aten.mm.default(view_581, permute_498);  permute_498 = None
    permute_499: "f32[256, 512]" = torch.ops.aten.permute.default(view_581, [1, 0])
    mm_133: "f32[256, 256]" = torch.ops.aten.mm.default(permute_499, view_24);  permute_499 = None
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_186: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
    view_582: "f32[256]" = torch.ops.aten.reshape.default(sum_186, [256]);  sum_186 = None
    permute_501: "f32[256, 256]" = torch.ops.aten.permute.default(permute_500, [1, 0]);  permute_500 = None
    view_583: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_132, [1, 512, 256]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_170: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_169, view_583);  add_169 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_584: "f32[512, 256]" = torch.ops.aten.reshape.default(view_575, [512, 256]);  view_575 = None
    mm_134: "f32[512, 256]" = torch.ops.aten.mm.default(view_584, permute_502);  permute_502 = None
    permute_503: "f32[256, 512]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_135: "f32[256, 256]" = torch.ops.aten.mm.default(permute_503, view_24);  permute_503 = view_24 = None
    permute_504: "f32[256, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_187: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
    view_585: "f32[256]" = torch.ops.aten.reshape.default(sum_187, [256]);  sum_187 = None
    permute_505: "f32[256, 256]" = torch.ops.aten.permute.default(permute_504, [1, 0]);  permute_504 = None
    view_586: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_134, [1, 512, 256]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_171: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_170, view_586);  add_170 = view_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:442, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_428: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, primals_22);  primals_22 = None
    mul_429: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_428, 256)
    sum_188: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True)
    mul_430: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_428, mul_8);  mul_428 = None
    sum_189: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True);  mul_430 = None
    mul_431: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_8, sum_189);  sum_189 = None
    sub_123: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_429, sum_188);  mul_429 = sum_188 = None
    sub_124: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_123, mul_431);  sub_123 = mul_431 = None
    mul_432: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_60, sub_124);  div_60 = sub_124 = None
    mul_433: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_171, mul_8);  mul_8 = None
    sum_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 1]);  mul_433 = None
    sum_191: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:441, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_34: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_434: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
    mul_435: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_432, mul_434);  mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:440, code: hidden_states = self.dense(hidden_states)
    view_587: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_435, [512, 256]);  mul_435 = None
    mm_136: "f32[512, 1024]" = torch.ops.aten.mm.default(view_587, permute_506);  permute_506 = None
    permute_507: "f32[256, 512]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_137: "f32[256, 1024]" = torch.ops.aten.mm.default(permute_507, view_22);  permute_507 = view_22 = None
    permute_508: "f32[1024, 256]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_192: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[256]" = torch.ops.aten.reshape.default(sum_192, [256]);  sum_192 = None
    permute_509: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_589: "f32[1, 512, 1024]" = torch.ops.aten.reshape.default(mm_136, [1, 512, 1024]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_437: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
    mul_438: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, view_21)
    mul_439: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_438, -0.5);  mul_438 = None
    exp_26: "f32[1, 512, 1024]" = torch.ops.aten.exp.default(mul_439);  mul_439 = None
    mul_440: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_441: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_21, mul_440);  view_21 = mul_440 = None
    add_173: "f32[1, 512, 1024]" = torch.ops.aten.add.Tensor(mul_437, mul_441);  mul_437 = mul_441 = None
    mul_442: "f32[1, 512, 1024]" = torch.ops.aten.mul.Tensor(view_589, add_173);  view_589 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    view_590: "f32[512, 1024]" = torch.ops.aten.reshape.default(mul_442, [512, 1024]);  mul_442 = None
    mm_138: "f32[512, 256]" = torch.ops.aten.mm.default(view_590, permute_510);  permute_510 = None
    permute_511: "f32[1024, 512]" = torch.ops.aten.permute.default(view_590, [1, 0])
    mm_139: "f32[1024, 256]" = torch.ops.aten.mm.default(permute_511, view_20);  permute_511 = view_20 = None
    permute_512: "f32[256, 1024]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_193: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
    view_591: "f32[1024]" = torch.ops.aten.reshape.default(sum_193, [1024]);  sum_193 = None
    permute_513: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_592: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_138, [1, 512, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:426, code: hidden_states = self.dense(hidden_states)
    add_174: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_432, view_592);  mul_432 = view_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:361, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_444: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_174, primals_16);  primals_16 = None
    mul_445: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_444, 256)
    sum_194: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_444, [2], True)
    mul_446: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_444, mul_3);  mul_444 = None
    sum_195: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_446, [2], True);  mul_446 = None
    mul_447: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_3, sum_195);  sum_195 = None
    sub_126: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(mul_445, sum_194);  mul_445 = sum_194 = None
    sub_127: "f32[1, 512, 256]" = torch.ops.aten.sub.Tensor(sub_126, mul_447);  sub_126 = mul_447 = None
    mul_448: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(div_61, sub_127);  div_61 = sub_127 = None
    mul_449: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(add_174, mul_3);  mul_3 = None
    sum_196: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 1]);  mul_449 = None
    sum_197: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_174, [0, 1]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:360, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_35: "f32[1, 512, 256]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_450: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
    mul_451: "f32[1, 512, 256]" = torch.ops.aten.mul.Tensor(mul_448, mul_450);  mul_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:359, code: hidden_states = self.dense(hidden_states)
    view_593: "f32[512, 256]" = torch.ops.aten.reshape.default(mul_451, [512, 256]);  mul_451 = None
    mm_140: "f32[512, 256]" = torch.ops.aten.mm.default(view_593, permute_514);  permute_514 = None
    permute_515: "f32[256, 512]" = torch.ops.aten.permute.default(view_593, [1, 0])
    mm_141: "f32[256, 256]" = torch.ops.aten.mm.default(permute_515, view_18);  permute_515 = view_18 = None
    permute_516: "f32[256, 256]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_198: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
    view_594: "f32[256]" = torch.ops.aten.reshape.default(sum_198, [256]);  sum_198 = None
    permute_517: "f32[256, 256]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_595: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_140, [1, 512, 256]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:341, code: context_layer = context_layer.view(new_context_layer_shape)
    view_596: "f32[1, 512, 4, 64]" = torch.ops.aten.reshape.default(view_595, [1, 512, 4, 64]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:339, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    permute_518: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None
    
    # No stacktrace found for following nodes
    view_default_138: "f32[4, 512, 64]" = torch.ops.aten.reshape.default(permute_518, [4, 512, 64]);  permute_518 = None
    bmm_default_68: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(permute_default_67, view_default_138);  permute_default_67 = None
    view_default_139: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_68, [1, 4, 512, 64]);  bmm_default_68 = None
    bmm_default_69: "f32[4, 512, 512]" = torch.ops.aten.bmm.default(view_default_138, permute_default_68);  view_default_138 = permute_default_68 = None
    view_default_140: "f32[1, 4, 512, 512]" = torch.ops.aten.reshape.default(bmm_default_69, [1, 4, 512, 512]);  bmm_default_69 = None
    mul_tensor_45: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(view_default_140, mul_tensor_44);  view_default_140 = mul_tensor_44 = None
    mul_tensor_46: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(mul_tensor_45, alias_default_23);  mul_tensor_45 = None
    sum_dim_int_list_23: "f32[1, 4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [-1], True)
    mul_tensor_47: "f32[1, 4, 512, 512]" = torch.ops.aten.mul.Tensor(alias_default_23, sum_dim_int_list_23);  alias_default_23 = sum_dim_int_list_23 = None
    sub_tensor_23: "f32[1, 4, 512, 512]" = torch.ops.aten.sub.Tensor(mul_tensor_46, mul_tensor_47);  mul_tensor_46 = mul_tensor_47 = None
    view_default_141: "f32[4, 512, 512]" = torch.ops.aten.reshape.default(sub_tensor_23, [4, 512, 512]);  sub_tensor_23 = None
    bmm_default_70: "f32[4, 64, 512]" = torch.ops.aten.bmm.default(permute_default_69, view_default_141);  permute_default_69 = None
    view_default_142: "f32[1, 4, 64, 512]" = torch.ops.aten.reshape.default(bmm_default_70, [1, 4, 64, 512]);  bmm_default_70 = None
    mul_scalar_46: "f32[1, 4, 64, 512]" = torch.ops.aten.mul.Scalar(view_default_142, 0.3535533905932738);  view_default_142 = None
    permute_default_71: "f32[1, 4, 512, 64]" = torch.ops.aten.permute.default(mul_scalar_46, [0, 1, 3, 2]);  mul_scalar_46 = None
    bmm_default_71: "f32[4, 512, 64]" = torch.ops.aten.bmm.default(view_default_141, permute_default_70);  view_default_141 = permute_default_70 = None
    view_default_143: "f32[1, 4, 512, 64]" = torch.ops.aten.reshape.default(bmm_default_71, [1, 4, 512, 64]);  bmm_default_71 = None
    mul_scalar_47: "f32[1, 4, 512, 64]" = torch.ops.aten.mul.Scalar(view_default_143, 0.3535533905932738);  view_default_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_524: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(mul_scalar_47, [0, 2, 1, 3]);  mul_scalar_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_70: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_524, memory_format = torch.contiguous_format);  permute_524 = None
    view_603: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_70, [1, 512, 256]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_525: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(view_default_139, [0, 2, 1, 3]);  view_default_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    clone_71: "f32[1, 512, 4, 64]" = torch.ops.aten.clone.default(permute_525, memory_format = torch.contiguous_format);  permute_525 = None
    view_604: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(clone_71, [1, 512, 256]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    view_605: "f32[512, 256]" = torch.ops.aten.reshape.default(view_604, [512, 256]);  view_604 = None
    mm_142: "f32[512, 256]" = torch.ops.aten.mm.default(view_605, permute_526);  permute_526 = None
    permute_527: "f32[256, 512]" = torch.ops.aten.permute.default(view_605, [1, 0])
    mm_143: "f32[256, 256]" = torch.ops.aten.mm.default(permute_527, view_2);  permute_527 = None
    permute_528: "f32[256, 256]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
    view_606: "f32[256]" = torch.ops.aten.reshape.default(sum_200, [256]);  sum_200 = None
    permute_529: "f32[256, 256]" = torch.ops.aten.permute.default(permute_528, [1, 0]);  permute_528 = None
    view_607: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_142, [1, 512, 256]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:281, code: value_layer = self.transpose_for_scores(self.value(hidden_states))
    add_175: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(mul_448, view_607);  mul_448 = view_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:246, code: return x.permute(0, 2, 1, 3)
    permute_530: "f32[1, 512, 4, 64]" = torch.ops.aten.permute.default(permute_default_71, [0, 2, 1, 3]);  permute_default_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:245, code: x = x.view(new_x_shape)
    view_608: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(permute_530, [1, 512, 256]);  permute_530 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    view_609: "f32[512, 256]" = torch.ops.aten.reshape.default(view_608, [512, 256]);  view_608 = None
    mm_144: "f32[512, 256]" = torch.ops.aten.mm.default(view_609, permute_531);  permute_531 = None
    permute_532: "f32[256, 512]" = torch.ops.aten.permute.default(view_609, [1, 0])
    mm_145: "f32[256, 256]" = torch.ops.aten.mm.default(permute_532, view_2);  permute_532 = None
    permute_533: "f32[256, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_201: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
    view_610: "f32[256]" = torch.ops.aten.reshape.default(sum_201, [256]);  sum_201 = None
    permute_534: "f32[256, 256]" = torch.ops.aten.permute.default(permute_533, [1, 0]);  permute_533 = None
    view_611: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_144, [1, 512, 256]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:280, code: key_layer = self.transpose_for_scores(self.key(hidden_states))
    add_176: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_175, view_611);  add_175 = view_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    view_612: "f32[512, 256]" = torch.ops.aten.reshape.default(view_603, [512, 256]);  view_603 = None
    mm_146: "f32[512, 256]" = torch.ops.aten.mm.default(view_612, permute_535);  permute_535 = None
    permute_536: "f32[256, 512]" = torch.ops.aten.permute.default(view_612, [1, 0])
    mm_147: "f32[256, 256]" = torch.ops.aten.mm.default(permute_536, view_2);  permute_536 = view_2 = None
    permute_537: "f32[256, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_202: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
    view_613: "f32[256]" = torch.ops.aten.reshape.default(sum_202, [256]);  sum_202 = None
    permute_538: "f32[256, 256]" = torch.ops.aten.permute.default(permute_537, [1, 0]);  permute_537 = None
    view_614: "f32[1, 512, 256]" = torch.ops.aten.reshape.default(mm_146, [1, 512, 256]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:258, code: mixed_query_layer = self.query(hidden_states)
    add_177: "f32[1, 512, 256]" = torch.ops.aten.add.Tensor(add_176, view_614);  add_176 = view_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:918, code: hidden_states = self.embeddings_project(hidden_states)
    view_615: "f32[512, 256]" = torch.ops.aten.reshape.default(add_177, [512, 256]);  add_177 = None
    mm_148: "f32[512, 128]" = torch.ops.aten.mm.default(view_615, permute_539);  permute_539 = None
    permute_540: "f32[256, 512]" = torch.ops.aten.permute.default(view_615, [1, 0])
    mm_149: "f32[256, 128]" = torch.ops.aten.mm.default(permute_540, view);  permute_540 = view = None
    permute_541: "f32[128, 256]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_203: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
    view_616: "f32[256]" = torch.ops.aten.reshape.default(sum_203, [256]);  sum_203 = None
    permute_542: "f32[256, 128]" = torch.ops.aten.permute.default(permute_541, [1, 0]);  permute_541 = None
    view_617: "f32[1, 512, 128]" = torch.ops.aten.reshape.default(mm_148, [1, 512, 128]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:211, code: embeddings = self.dropout(embeddings)
    convert_element_type_37: "f32[1, 512, 128]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_456: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
    mul_457: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(view_617, mul_456);  view_617 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:210, code: embeddings = self.LayerNorm(embeddings)
    mul_459: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_457, primals_4);  primals_4 = None
    mul_460: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, 128)
    sum_204: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
    mul_461: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_459, mul_1);  mul_459 = None
    sum_205: "f32[1, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
    mul_462: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_1, sum_205);  sum_205 = None
    sub_130: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(mul_460, sum_204);  mul_460 = sum_204 = None
    sub_131: "f32[1, 512, 128]" = torch.ops.aten.sub.Tensor(sub_130, mul_462);  sub_130 = mul_462 = None
    mul_463: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(div_63, sub_131);  div_63 = sub_131 = None
    mul_464: "f32[1, 512, 128]" = torch.ops.aten.mul.Tensor(mul_457, mul_1);  mul_1 = None
    sum_206: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
    sum_207: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 1]);  mul_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:208, code: position_embeddings = self.position_embeddings(position_ids)
    eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(slice_4, -1)
    unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
    where_4: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_4, full_default_2, mul_463);  unsqueeze_4 = None
    full_default_10: "f32[512, 128]" = torch.ops.aten.full.default([512, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[512, 128]" = torch.ops.prims._unsafe_index_put_.default(full_default_10, [slice_4], where_4, True);  full_default_10 = slice_4 = where_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:204, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
    eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(expand, -1)
    unsqueeze_5: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
    where_5: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_5, full_default_2, mul_463);  unsqueeze_5 = None
    full_default_12: "f32[2, 128]" = torch.ops.aten.full.default([2, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[2, 128]" = torch.ops.prims._unsafe_index_put_.default(full_default_12, [expand], where_5, True);  full_default_12 = expand = where_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/electra/modeling_electra.py:203, code: inputs_embeds = self.word_embeddings(input_ids)
    eq_2: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_209, 0)
    unsqueeze_6: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
    where_6: "f32[1, 512, 128]" = torch.ops.aten.where.self(unsqueeze_6, full_default_2, mul_463);  unsqueeze_6 = full_default_2 = mul_463 = None
    full_default_14: "f32[30522, 128]" = torch.ops.aten.full.default([30522, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_2: "f32[30522, 128]" = torch.ops.prims._unsafe_index_put_.default(full_default_14, [primals_209], where_6, True);  full_default_14 = primals_209 = where_6 = None
    return [_unsafe_index_put_2, _unsafe_index_put_1, _unsafe_index_put, sum_206, sum_207, permute_542, view_616, permute_538, view_613, permute_534, view_610, permute_529, view_606, permute_517, view_594, sum_196, sum_197, permute_513, view_591, permute_509, view_588, sum_190, sum_191, permute_505, view_585, permute_501, view_582, permute_496, view_578, permute_484, view_566, sum_181, sum_182, permute_480, view_563, permute_476, view_560, sum_175, sum_176, permute_472, view_557, permute_468, view_554, permute_463, view_550, permute_451, view_538, sum_166, sum_167, permute_447, view_535, permute_443, view_532, sum_160, sum_161, permute_439, view_529, permute_435, view_526, permute_430, view_522, permute_418, view_510, sum_151, sum_152, permute_414, view_507, permute_410, view_504, sum_145, sum_146, permute_406, view_501, permute_402, view_498, permute_397, view_494, permute_385, view_482, sum_136, sum_137, permute_381, view_479, permute_377, view_476, sum_130, sum_131, permute_373, view_473, permute_369, view_470, permute_364, view_466, permute_352, view_454, sum_121, sum_122, permute_348, view_451, permute_344, view_448, sum_115, sum_116, permute_340, view_445, permute_336, view_442, permute_331, view_438, permute_319, view_426, sum_106, sum_107, permute_315, view_423, permute_311, view_420, sum_100, sum_101, permute_307, view_417, permute_303, view_414, permute_298, view_410, permute_286, view_398, sum_91, sum_92, permute_282, view_395, permute_278, view_392, sum_85, sum_86, permute_274, view_389, permute_270, view_386, permute_265, view_382, permute_253, view_370, sum_76, sum_77, permute_249, view_367, permute_245, view_364, sum_70, sum_71, permute_241, view_361, permute_237, view_358, permute_232, view_354, permute_220, view_342, sum_61, sum_62, permute_216, view_339, permute_212, view_336, sum_55, sum_56, permute_208, view_333, permute_204, view_330, permute_199, view_326, permute_187, view_314, sum_46, sum_47, permute_183, view_311, permute_179, view_308, sum_40, sum_41, permute_175, view_305, permute_171, view_302, permute_166, view_298, permute_154, view_286, sum_31, sum_32, permute_150, view_283, permute_146, view_280, sum_25, sum_26, permute_142, view_277, sum_20, sum_21, permute_138, view_274, None, None, None, None]
    