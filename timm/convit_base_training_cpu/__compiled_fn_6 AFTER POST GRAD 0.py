from __future__ import annotations



def forward(self, primals_3: "f32[768]", primals_5: "f32[16]", primals_6: "f32[768]", primals_8: "f32[768]", primals_10: "f32[16]", primals_11: "f32[768]", primals_13: "f32[768]", primals_15: "f32[16]", primals_16: "f32[768]", primals_18: "f32[768]", primals_20: "f32[16]", primals_21: "f32[768]", primals_23: "f32[768]", primals_25: "f32[16]", primals_26: "f32[768]", primals_28: "f32[768]", primals_30: "f32[16]", primals_31: "f32[768]", primals_33: "f32[768]", primals_35: "f32[16]", primals_36: "f32[768]", primals_38: "f32[768]", primals_40: "f32[16]", primals_41: "f32[768]", primals_43: "f32[768]", primals_45: "f32[16]", primals_46: "f32[768]", primals_48: "f32[768]", primals_50: "f32[16]", primals_51: "f32[768]", primals_53: "f32[768]", primals_55: "f32[768]", primals_57: "f32[768]", primals_59: "f32[768]", primals_61: "f32[768]", primals_63: "f32[768, 3, 16, 16]", primals_181: "f32[8, 3, 224, 224]", mul: "f32[8, 196, 768]", view_5: "f32[1568, 768]", view_8: "f32[307328, 3]", div: "f32[8, 16, 196, 196]", div_1: "f32[8, 16, 196, 196]", unsqueeze_5: "f32[8, 16, 196, 1]", view_21: "f32[1568, 768]", mul_5: "f32[8, 196, 768]", view_23: "f32[1568, 768]", addmm_1: "f32[1568, 3072]", view_25: "f32[1568, 3072]", mul_10: "f32[8, 196, 768]", view_31: "f32[1568, 768]", div_3: "f32[8, 16, 196, 196]", div_4: "f32[8, 16, 196, 196]", unsqueeze_11: "f32[8, 16, 196, 1]", view_47: "f32[1568, 768]", mul_15: "f32[8, 196, 768]", view_49: "f32[1568, 768]", addmm_4: "f32[1568, 3072]", view_51: "f32[1568, 3072]", mul_20: "f32[8, 196, 768]", view_57: "f32[1568, 768]", div_6: "f32[8, 16, 196, 196]", div_7: "f32[8, 16, 196, 196]", unsqueeze_17: "f32[8, 16, 196, 1]", view_73: "f32[1568, 768]", mul_25: "f32[8, 196, 768]", view_75: "f32[1568, 768]", addmm_7: "f32[1568, 3072]", view_77: "f32[1568, 3072]", mul_30: "f32[8, 196, 768]", view_83: "f32[1568, 768]", div_9: "f32[8, 16, 196, 196]", div_10: "f32[8, 16, 196, 196]", unsqueeze_23: "f32[8, 16, 196, 1]", view_99: "f32[1568, 768]", mul_35: "f32[8, 196, 768]", view_101: "f32[1568, 768]", addmm_10: "f32[1568, 3072]", view_103: "f32[1568, 3072]", mul_40: "f32[8, 196, 768]", view_109: "f32[1568, 768]", div_12: "f32[8, 16, 196, 196]", div_13: "f32[8, 16, 196, 196]", unsqueeze_29: "f32[8, 16, 196, 1]", view_125: "f32[1568, 768]", mul_45: "f32[8, 196, 768]", view_127: "f32[1568, 768]", addmm_13: "f32[1568, 3072]", view_129: "f32[1568, 3072]", mul_50: "f32[8, 196, 768]", view_135: "f32[1568, 768]", div_15: "f32[8, 16, 196, 196]", div_16: "f32[8, 16, 196, 196]", unsqueeze_35: "f32[8, 16, 196, 1]", view_151: "f32[1568, 768]", mul_55: "f32[8, 196, 768]", view_153: "f32[1568, 768]", addmm_16: "f32[1568, 3072]", view_155: "f32[1568, 3072]", mul_60: "f32[8, 196, 768]", view_161: "f32[1568, 768]", div_18: "f32[8, 16, 196, 196]", div_19: "f32[8, 16, 196, 196]", unsqueeze_41: "f32[8, 16, 196, 1]", view_177: "f32[1568, 768]", mul_65: "f32[8, 196, 768]", view_179: "f32[1568, 768]", addmm_19: "f32[1568, 3072]", view_181: "f32[1568, 3072]", mul_70: "f32[8, 196, 768]", view_187: "f32[1568, 768]", div_21: "f32[8, 16, 196, 196]", div_22: "f32[8, 16, 196, 196]", unsqueeze_47: "f32[8, 16, 196, 1]", view_203: "f32[1568, 768]", mul_75: "f32[8, 196, 768]", view_205: "f32[1568, 768]", addmm_22: "f32[1568, 3072]", view_207: "f32[1568, 3072]", mul_80: "f32[8, 196, 768]", view_213: "f32[1568, 768]", div_24: "f32[8, 16, 196, 196]", div_25: "f32[8, 16, 196, 196]", unsqueeze_53: "f32[8, 16, 196, 1]", view_229: "f32[1568, 768]", mul_85: "f32[8, 196, 768]", view_231: "f32[1568, 768]", addmm_25: "f32[1568, 3072]", view_233: "f32[1568, 3072]", mul_90: "f32[8, 196, 768]", view_239: "f32[1568, 768]", div_27: "f32[8, 16, 196, 196]", div_28: "f32[8, 16, 196, 196]", unsqueeze_59: "f32[8, 16, 196, 1]", view_255: "f32[1568, 768]", mul_95: "f32[8, 196, 768]", view_257: "f32[1568, 768]", addmm_28: "f32[1568, 3072]", view_259: "f32[1568, 3072]", cat: "f32[8, 197, 768]", getitem_41: "f32[8, 197, 1]", rsqrt_20: "f32[8, 197, 1]", view_261: "f32[1576, 768]", view_271: "f32[1576, 768]", mul_103: "f32[8, 197, 768]", view_273: "f32[1576, 768]", addmm_31: "f32[1576, 3072]", view_275: "f32[1576, 3072]", mul_108: "f32[8, 197, 768]", view_277: "f32[1576, 768]", view_287: "f32[1576, 768]", mul_111: "f32[8, 197, 768]", view_289: "f32[1576, 768]", addmm_34: "f32[1576, 3072]", view_291: "f32[1576, 3072]", mul_116: "f32[8, 197, 768]", clone_167: "f32[8, 768]", permute_126: "f32[1000, 768]", div_32: "f32[8, 197, 1]", permute_130: "f32[768, 3072]", permute_134: "f32[3072, 768]", div_33: "f32[8, 197, 1]", permute_138: "f32[768, 768]", permute_143: "f32[128, 197, 197]", permute_144: "f32[128, 48, 197]", alias_42: "f32[8, 16, 197, 197]", permute_145: "f32[128, 48, 197]", permute_146: "f32[128, 197, 48]", permute_151: "f32[2304, 768]", div_34: "f32[8, 197, 1]", permute_153: "f32[768, 3072]", permute_157: "f32[3072, 768]", div_35: "f32[8, 197, 1]", permute_161: "f32[768, 768]", permute_166: "f32[128, 197, 197]", permute_167: "f32[128, 48, 197]", alias_43: "f32[8, 16, 197, 197]", permute_168: "f32[128, 48, 197]", permute_169: "f32[128, 197, 48]", permute_174: "f32[2304, 768]", permute_176: "f32[768, 3072]", permute_180: "f32[3072, 768]", div_37: "f32[8, 196, 1]", permute_184: "f32[768, 768]", permute_189: "f32[128, 196, 196]", permute_190: "f32[128, 48, 196]", permute_194: "f32[768, 768]", permute_196: "f32[128, 48, 196]", permute_197: "f32[128, 196, 48]", permute_206: "f32[1536, 768]", div_41: "f32[8, 196, 1]", permute_208: "f32[768, 3072]", permute_212: "f32[3072, 768]", div_42: "f32[8, 196, 1]", permute_216: "f32[768, 768]", permute_221: "f32[128, 196, 196]", permute_222: "f32[128, 48, 196]", permute_226: "f32[768, 768]", permute_228: "f32[128, 48, 196]", permute_229: "f32[128, 196, 48]", permute_238: "f32[1536, 768]", div_46: "f32[8, 196, 1]", permute_240: "f32[768, 3072]", permute_244: "f32[3072, 768]", div_47: "f32[8, 196, 1]", permute_248: "f32[768, 768]", permute_253: "f32[128, 196, 196]", permute_254: "f32[128, 48, 196]", permute_258: "f32[768, 768]", permute_260: "f32[128, 48, 196]", permute_261: "f32[128, 196, 48]", permute_270: "f32[1536, 768]", div_51: "f32[8, 196, 1]", permute_272: "f32[768, 3072]", permute_276: "f32[3072, 768]", div_52: "f32[8, 196, 1]", permute_280: "f32[768, 768]", permute_285: "f32[128, 196, 196]", permute_286: "f32[128, 48, 196]", permute_290: "f32[768, 768]", permute_292: "f32[128, 48, 196]", permute_293: "f32[128, 196, 48]", permute_302: "f32[1536, 768]", div_56: "f32[8, 196, 1]", permute_304: "f32[768, 3072]", permute_308: "f32[3072, 768]", div_57: "f32[8, 196, 1]", permute_312: "f32[768, 768]", permute_317: "f32[128, 196, 196]", permute_318: "f32[128, 48, 196]", permute_322: "f32[768, 768]", permute_324: "f32[128, 48, 196]", permute_325: "f32[128, 196, 48]", permute_334: "f32[1536, 768]", div_61: "f32[8, 196, 1]", permute_336: "f32[768, 3072]", permute_340: "f32[3072, 768]", div_62: "f32[8, 196, 1]", permute_344: "f32[768, 768]", permute_349: "f32[128, 196, 196]", permute_350: "f32[128, 48, 196]", permute_354: "f32[768, 768]", permute_356: "f32[128, 48, 196]", permute_357: "f32[128, 196, 48]", permute_366: "f32[1536, 768]", div_66: "f32[8, 196, 1]", permute_368: "f32[768, 3072]", permute_372: "f32[3072, 768]", div_67: "f32[8, 196, 1]", permute_376: "f32[768, 768]", permute_381: "f32[128, 196, 196]", permute_382: "f32[128, 48, 196]", permute_386: "f32[768, 768]", permute_388: "f32[128, 48, 196]", permute_389: "f32[128, 196, 48]", permute_398: "f32[1536, 768]", div_71: "f32[8, 196, 1]", permute_400: "f32[768, 3072]", permute_404: "f32[3072, 768]", div_72: "f32[8, 196, 1]", permute_408: "f32[768, 768]", permute_413: "f32[128, 196, 196]", permute_414: "f32[128, 48, 196]", permute_418: "f32[768, 768]", permute_420: "f32[128, 48, 196]", permute_421: "f32[128, 196, 48]", permute_430: "f32[1536, 768]", div_76: "f32[8, 196, 1]", permute_432: "f32[768, 3072]", permute_436: "f32[3072, 768]", div_77: "f32[8, 196, 1]", permute_440: "f32[768, 768]", permute_445: "f32[128, 196, 196]", permute_446: "f32[128, 48, 196]", permute_450: "f32[768, 768]", permute_452: "f32[128, 48, 196]", permute_453: "f32[128, 196, 48]", permute_462: "f32[1536, 768]", div_81: "f32[8, 196, 1]", permute_464: "f32[768, 3072]", permute_468: "f32[3072, 768]", div_82: "f32[8, 196, 1]", permute_472: "f32[768, 768]", permute_477: "f32[128, 196, 196]", permute_478: "f32[128, 48, 196]", permute_482: "f32[768, 768]", permute_484: "f32[128, 48, 196]", permute_485: "f32[128, 196, 48]", permute_494: "f32[1536, 768]", div_86: "f32[8, 196, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_13: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_5, [1, -1, 1, 1]);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_13);  view_13 = None
    sub_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid)
    mul_3: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_4, div)
    mul_4: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid, div_1)
    add_5: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_24: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_1, [8, 196, 3072]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_8: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, 0.7071067811865476)
    erf: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_9: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_39: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_10, [1, -1, 1, 1]);  primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_2: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_39);  view_39 = None
    sub_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_2)
    mul_13: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_10, div_3)
    mul_14: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_2, div_4)
    add_15: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_13, mul_14);  mul_13 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_50: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_4, [8, 196, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476)
    erf_1: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_19: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_65: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_15, [1, -1, 1, 1]);  primals_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_4: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_65);  view_65 = None
    sub_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_4)
    mul_23: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_16, div_6)
    mul_24: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_4, div_7)
    add_25: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_76: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_7, [8, 196, 3072]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_28: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, 0.7071067811865476)
    erf_2: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_29: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_91: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_20, [1, -1, 1, 1]);  primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_6: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_91);  view_91 = None
    sub_22: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_6)
    mul_33: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_22, div_9)
    mul_34: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_6, div_10)
    add_35: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 196, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_3: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_39: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_117: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_25, [1, -1, 1, 1]);  primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_8: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_117);  view_117 = None
    sub_28: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_8)
    mul_43: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_28, div_12)
    mul_44: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_8, div_13)
    add_45: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_128: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_13, [8, 196, 3072]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_48: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, 0.7071067811865476)
    erf_4: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_49: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_143: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_30, [1, -1, 1, 1]);  primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_10: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_143);  view_143 = None
    sub_34: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_10)
    mul_53: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_34, div_15)
    mul_54: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_10, div_16)
    add_55: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_53, mul_54);  mul_53 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_154: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_16, [8, 196, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_58: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, 0.7071067811865476)
    erf_5: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_59: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_169: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_35, [1, -1, 1, 1]);  primals_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_12: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_169);  view_169 = None
    sub_40: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_12)
    mul_63: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_40, div_18)
    mul_64: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_12, div_19)
    add_65: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_19, [8, 196, 3072]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, 0.7071067811865476)
    erf_6: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_195: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_40, [1, -1, 1, 1]);  primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_14: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_195);  view_195 = None
    sub_46: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_14)
    mul_73: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_46, div_21)
    mul_74: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_14, div_22)
    add_75: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_206: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 196, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_78: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, 0.7071067811865476)
    erf_7: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_79: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_221: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_45, [1, -1, 1, 1]);  primals_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_16: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_221);  view_221 = None
    sub_52: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_16)
    mul_83: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_52, div_24)
    mul_84: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_16, div_25)
    add_85: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_232: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_25, [8, 196, 3072]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_88: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, 0.7071067811865476)
    erf_8: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_89: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_247: "f32[1, 16, 1, 1]" = torch.ops.aten.reshape.default(primals_50, [1, -1, 1, 1]);  primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    sigmoid_18: "f32[1, 16, 1, 1]" = torch.ops.aten.sigmoid.default(view_247);  view_247 = None
    sub_58: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1.0, sigmoid_18)
    mul_93: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_58, div_27)
    mul_94: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sigmoid_18, div_28)
    add_95: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(mul_93, mul_94);  mul_93 = mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_258: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(addmm_28, [8, 196, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_98: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, 0.7071067811865476)
    erf_9: "f32[8, 196, 3072]" = torch.ops.aten.erf.default(mul_98);  mul_98 = None
    add_99: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_41);  cat = getitem_41 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_20);  sub_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_31, [8, 197, 3072]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_290: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_113: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:376, code: return x if pre_logits else self.head(x)
    mm_32: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_126);  permute_126 = None
    permute_127: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_33: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_127, clone_167);  permute_127 = clone_167 = None
    permute_128: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_33: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_293: "f32[1000]" = torch.ops.aten.reshape.default(sum_33, [1000]);  sum_33 = None
    permute_129: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:374, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full_default: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_30: "f32[8, 197, 768]" = torch.ops.aten.select_scatter.default(full_default, mm_32, 1, 0);  full_default = mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_119: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(select_scatter_30, primals_61);  primals_61 = None
    mul_120: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_119, 768)
    sum_34: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True)
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_119, mul_116);  mul_119 = None
    sum_35: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_116, sum_35);  sum_35 = None
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_120, sum_34);  mul_120 = sum_34 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_122);  sub_68 = mul_122 = None
    mul_123: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_69);  div_32 = sub_69 = None
    mul_124: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(select_scatter_30, mul_116);  mul_116 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1]);  mul_124 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(select_scatter_30, [0, 1]);  select_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_294: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_123, [1576, 768])
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_294, permute_130);  permute_130 = None
    permute_131: "f32[768, 1576]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_131, view_291);  permute_131 = view_291 = None
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
    permute_133: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    view_296: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_126: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_113, 0.5);  add_113 = None
    mul_127: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, view_290)
    mul_128: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_127, -0.5);  mul_127 = None
    exp_22: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_128);  mul_128 = None
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_130: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_290, mul_129);  view_290 = mul_129 = None
    add_118: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_126, mul_130);  mul_126 = mul_130 = None
    mul_131: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_296, add_118);  view_296 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_297: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_131, [1576, 3072]);  mul_131 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_297, permute_134);  permute_134 = None
    permute_135: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_297, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_135, view_289);  permute_135 = view_289 = None
    permute_136: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_39: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
    view_298: "f32[3072]" = torch.ops.aten.reshape.default(sum_39, [3072]);  sum_39 = None
    permute_137: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_299: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_133: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_299, primals_59);  primals_59 = None
    mul_134: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_133, 768)
    sum_40: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_133, [2], True)
    mul_135: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_133, mul_111);  mul_133 = None
    sum_41: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True);  mul_135 = None
    mul_136: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_111, sum_41);  sum_41 = None
    sub_71: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_134, sum_40);  mul_134 = sum_40 = None
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_136);  sub_71 = mul_136 = None
    mul_137: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_72);  div_33 = sub_72 = None
    mul_138: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_299, mul_111);  mul_111 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_138, [0, 1]);  mul_138 = None
    sum_43: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_299, [0, 1]);  view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_119: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_123, mul_137);  mul_123 = mul_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_300: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_119, [1576, 768])
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_300, permute_138);  permute_138 = None
    permute_139: "f32[768, 1576]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_139, view_287);  permute_139 = view_287 = None
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_44: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[768]" = torch.ops.aten.reshape.default(sum_44, [768]);  sum_44 = None
    permute_141: "f32[768, 768]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    view_302: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_303: "f32[8, 197, 16, 48]" = torch.ops.aten.reshape.default(view_302, [8, 197, 16, 48]);  view_302 = None
    permute_142: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    clone_168: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_304: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_168, [128, 197, 48]);  clone_168 = None
    bmm_24: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(permute_143, view_304);  permute_143 = None
    bmm_25: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_304, permute_144);  view_304 = permute_144 = None
    view_305: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_24, [8, 16, 197, 48]);  bmm_24 = None
    view_306: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_25, [8, 16, 197, 197]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    mul_139: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_306, alias_42);  view_306 = None
    sum_45: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_139, [-1], True)
    mul_140: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(alias_42, sum_45);  alias_42 = sum_45 = None
    sub_73: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_141: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(sub_73, 0.14433756729740643);  sub_73 = None
    view_307: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(mul_141, [128, 197, 197]);  mul_141 = None
    bmm_26: "f32[128, 48, 197]" = torch.ops.aten.bmm.default(permute_145, view_307);  permute_145 = None
    bmm_27: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_307, permute_146);  view_307 = permute_146 = None
    view_308: "f32[8, 16, 48, 197]" = torch.ops.aten.reshape.default(bmm_26, [8, 16, 48, 197]);  bmm_26 = None
    view_309: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_27, [8, 16, 197, 48]);  bmm_27 = None
    permute_147: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_308, [0, 1, 3, 2]);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 16, 197, 48]" = torch.ops.aten.cat.default([view_309, permute_147, view_305]);  view_309 = permute_147 = view_305 = None
    view_310: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.reshape.default(cat_1, [3, 8, 16, 197, 48]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_148: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.permute.default(view_310, [1, 3, 0, 2, 4]);  view_310 = None
    clone_169: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_311: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_169, [8, 197, 2304]);  clone_169 = None
    view_312: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_311, [1576, 2304]);  view_311 = None
    permute_149: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_312, [1, 0])
    mm_40: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_149, view_277);  permute_149 = view_277 = None
    permute_150: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_40, [1, 0]);  mm_40 = None
    mm_41: "f32[1576, 768]" = torch.ops.aten.mm.default(view_312, permute_151);  view_312 = permute_151 = None
    view_313: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_41, [8, 197, 768]);  mm_41 = None
    permute_152: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_57);  primals_57 = None
    mul_144: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_46: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_108);  mul_143 = None
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, sum_47);  sum_47 = None
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_46);  mul_144 = sum_46 = None
    sub_76: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_75, mul_146);  sub_75 = mul_146 = None
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_76);  div_34 = sub_76 = None
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_108);  mul_108 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_119, mul_147);  add_119 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_314: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_120, [1576, 768])
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_314, permute_153);  permute_153 = None
    permute_154: "f32[768, 1576]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_154, view_275);  permute_154 = view_275 = None
    permute_155: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_50: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[768]" = torch.ops.aten.reshape.default(sum_50, [768]);  sum_50 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
    view_316: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_150: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_151: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, view_274)
    mul_152: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_151, -0.5);  mul_151 = None
    exp_23: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_152);  mul_152 = None
    mul_153: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_154: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, mul_153);  view_274 = mul_153 = None
    add_122: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_150, mul_154);  mul_150 = mul_154 = None
    mul_155: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_316, add_122);  view_316 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_317: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_155, [1576, 3072]);  mul_155 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_317, permute_157);  permute_157 = None
    permute_158: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_158, view_273);  permute_158 = view_273 = None
    permute_159: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_51: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[3072]" = torch.ops.aten.reshape.default(sum_51, [3072]);  sum_51 = None
    permute_160: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_319: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, primals_55);  primals_55 = None
    mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_157, 768)
    sum_52: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True)
    mul_159: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_157, mul_103);  mul_157 = None
    sum_53: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    mul_160: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_103, sum_53);  sum_53 = None
    sub_78: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_158, sum_52);  mul_158 = sum_52 = None
    sub_79: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_78, mul_160);  sub_78 = mul_160 = None
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_79);  div_35 = sub_79 = None
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, mul_103);  mul_103 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_162, [0, 1]);  mul_162 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_319, [0, 1]);  view_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_123: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_120, mul_161);  add_120 = mul_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:183, code: x = self.proj(x)
    view_320: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_123, [1576, 768])
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_320, permute_161);  permute_161 = None
    permute_162: "f32[768, 1576]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_271);  permute_162 = view_271 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[768]" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_322: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:182, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_323: "f32[8, 197, 16, 48]" = torch.ops.aten.reshape.default(view_322, [8, 197, 16, 48]);  view_322 = None
    permute_165: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
    clone_170: "f32[8, 16, 197, 48]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
    view_324: "f32[128, 197, 48]" = torch.ops.aten.reshape.default(clone_170, [128, 197, 48]);  clone_170 = None
    bmm_28: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(permute_166, view_324);  permute_166 = None
    bmm_29: "f32[128, 197, 197]" = torch.ops.aten.bmm.default(view_324, permute_167);  view_324 = permute_167 = None
    view_325: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_28, [8, 16, 197, 48]);  bmm_28 = None
    view_326: "f32[8, 16, 197, 197]" = torch.ops.aten.reshape.default(bmm_29, [8, 16, 197, 197]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:179, code: attn = attn.softmax(dim=-1)
    mul_163: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(view_326, alias_43);  view_326 = None
    sum_57: "f32[8, 16, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [-1], True)
    mul_164: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(alias_43, sum_57);  alias_43 = sum_57 = None
    sub_80: "f32[8, 16, 197, 197]" = torch.ops.aten.sub.Tensor(mul_163, mul_164);  mul_163 = mul_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:178, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    mul_165: "f32[8, 16, 197, 197]" = torch.ops.aten.mul.Tensor(sub_80, 0.14433756729740643);  sub_80 = None
    view_327: "f32[128, 197, 197]" = torch.ops.aten.reshape.default(mul_165, [128, 197, 197]);  mul_165 = None
    bmm_30: "f32[128, 48, 197]" = torch.ops.aten.bmm.default(permute_168, view_327);  permute_168 = None
    bmm_31: "f32[128, 197, 48]" = torch.ops.aten.bmm.default(view_327, permute_169);  view_327 = permute_169 = None
    view_328: "f32[8, 16, 48, 197]" = torch.ops.aten.reshape.default(bmm_30, [8, 16, 48, 197]);  bmm_30 = None
    view_329: "f32[8, 16, 197, 48]" = torch.ops.aten.reshape.default(bmm_31, [8, 16, 197, 48]);  bmm_31 = None
    permute_170: "f32[8, 16, 197, 48]" = torch.ops.aten.permute.default(view_328, [0, 1, 3, 2]);  view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:176, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 16, 197, 48]" = torch.ops.aten.cat.default([view_329, permute_170, view_325]);  view_329 = permute_170 = view_325 = None
    view_330: "f32[3, 8, 16, 197, 48]" = torch.ops.aten.reshape.default(cat_2, [3, 8, 16, 197, 48]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:175, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_171: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.permute.default(view_330, [1, 3, 0, 2, 4]);  view_330 = None
    clone_171: "f32[8, 197, 3, 16, 48]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_331: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(clone_171, [8, 197, 2304]);  clone_171 = None
    view_332: "f32[1576, 2304]" = torch.ops.aten.reshape.default(view_331, [1576, 2304]);  view_331 = None
    permute_172: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_48: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_172, view_261);  permute_172 = view_261 = None
    permute_173: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_48, [1, 0]);  mm_48 = None
    mm_49: "f32[1576, 768]" = torch.ops.aten.mm.default(view_332, permute_174);  view_332 = permute_174 = None
    view_333: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_49, [8, 197, 768]);  mm_49 = None
    permute_175: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_333, primals_53);  primals_53 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_167, 768)
    sum_58: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_167, mul_100);  mul_167 = None
    sum_59: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
    mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_59);  sum_59 = None
    sub_82: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_168, sum_58);  mul_168 = sum_58 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_82, mul_170);  sub_82 = mul_170 = None
    div_36: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_83);  div_36 = sub_83 = None
    mul_172: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_333, mul_100);  mul_100 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_123, mul_171);  add_123 = mul_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:367, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_332: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_124, 1, 0, 1)
    slice_333: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_124, 1, 1, 197);  add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    clone_172: "f32[8, 196, 768]" = torch.ops.aten.clone.default(slice_333, memory_format = torch.contiguous_format)
    view_334: "f32[1568, 768]" = torch.ops.aten.reshape.default(clone_172, [1568, 768]);  clone_172 = None
    mm_50: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_334, permute_176);  permute_176 = None
    permute_177: "f32[768, 1568]" = torch.ops.aten.permute.default(view_334, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_177, view_259);  permute_177 = view_259 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
    view_335: "f32[768]" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_336: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_50, [8, 196, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_174: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_99, 0.5);  add_99 = None
    mul_175: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, view_258)
    mul_176: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_175, -0.5);  mul_175 = None
    exp_24: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_176);  mul_176 = None
    mul_177: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_178: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_258, mul_177);  view_258 = mul_177 = None
    add_126: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_174, mul_178);  mul_174 = mul_178 = None
    mul_179: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_336, add_126);  view_336 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_337: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_179, [1568, 3072]);  mul_179 = None
    mm_52: "f32[1568, 768]" = torch.ops.aten.mm.default(view_337, permute_180);  permute_180 = None
    permute_181: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_337, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_181, view_257);  permute_181 = view_257 = None
    permute_182: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
    view_338: "f32[3072]" = torch.ops.aten.reshape.default(sum_63, [3072]);  sum_63 = None
    permute_183: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    view_339: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_52, [8, 196, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_181: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_339, primals_51);  primals_51 = None
    mul_182: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, 768)
    sum_64: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [2], True)
    mul_183: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_181, mul_95);  mul_181 = None
    sum_65: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [2], True);  mul_183 = None
    mul_184: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_95, sum_65);  sum_65 = None
    sub_85: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_182, sum_64);  mul_182 = sum_64 = None
    sub_86: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_85, mul_184);  sub_85 = mul_184 = None
    mul_185: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_86);  div_37 = sub_86 = None
    mul_186: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_339, mul_95);  mul_95 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_186, [0, 1]);  mul_186 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_127: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(slice_333, mul_185);  slice_333 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_340: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_127, [1568, 768])
    mm_54: "f32[1568, 768]" = torch.ops.aten.mm.default(view_340, permute_184);  permute_184 = None
    permute_185: "f32[768, 1568]" = torch.ops.aten.permute.default(view_340, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_255);  permute_185 = view_255 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True);  view_340 = None
    view_341: "f32[768]" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_342: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_54, [8, 196, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_343: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_342, [8, 196, 16, 48]);  view_342 = None
    permute_188: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    clone_174: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
    view_344: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_174, [128, 196, 48]);  clone_174 = None
    bmm_32: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_189, view_344);  permute_189 = None
    bmm_33: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_344, permute_190);  view_344 = permute_190 = None
    view_345: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_32, [8, 16, 196, 48]);  bmm_32 = None
    view_346: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_33, [8, 16, 196, 196]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_191: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_175: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    view_347: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_175, [8, 196, 768]);  clone_175 = None
    view_348: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_347, [1568, 768]);  view_347 = None
    permute_192: "f32[768, 1568]" = torch.ops.aten.permute.default(view_348, [1, 0])
    mm_56: "f32[768, 768]" = torch.ops.aten.mm.default(permute_192, view_239);  permute_192 = None
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(mm_56, [1, 0]);  mm_56 = None
    mm_57: "f32[1568, 768]" = torch.ops.aten.mm.default(view_348, permute_194);  view_348 = permute_194 = None
    view_349: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_57, [8, 196, 768]);  mm_57 = None
    permute_195: "f32[768, 768]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_38: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_95, unsqueeze_59);  add_95 = None
    div_39: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_38, unsqueeze_59);  div_38 = None
    neg: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_346)
    mul_187: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg, div_39);  neg = div_39 = None
    div_40: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_346, unsqueeze_59);  view_346 = unsqueeze_59 = None
    sum_69: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [3], True);  mul_187 = None
    squeeze: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_69, -1);  sum_69 = None
    unsqueeze_60: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze, -1);  squeeze = None
    expand_89: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_60, [8, 16, 196, 196]);  unsqueeze_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_128: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_40, expand_89);  div_40 = expand_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_188: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, sigmoid_18)
    mul_189: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, div_28)
    sum_70: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3], True);  mul_189 = None
    sub_87: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_18)
    mul_190: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_18, sub_87);  sigmoid_18 = sub_87 = None
    mul_191: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_70, mul_190);  sum_70 = None
    mul_192: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, sub_58);  sub_58 = None
    mul_193: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_128, div_27);  add_128 = None
    sum_71: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 2, 3], True);  mul_193 = None
    neg_1: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_71);  sum_71 = None
    mul_195: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_1, mul_190);  neg_1 = mul_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_129: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_191, mul_195);  mul_191 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_350: "f32[16]" = torch.ops.aten.reshape.default(add_129, [16]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_196: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_188, div_28);  mul_188 = None
    sum_72: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [-1], True)
    mul_197: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_28, sum_72);  div_28 = sum_72 = None
    sub_89: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_198: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_192, div_27);  mul_192 = None
    sum_73: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_198, [-1], True)
    mul_199: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_27, sum_73);  div_27 = sum_73 = None
    sub_90: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_200: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_90, 0.14433756729740643);  sub_90 = None
    view_351: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_200, [128, 196, 196]);  mul_200 = None
    bmm_34: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_196, view_351);  permute_196 = None
    bmm_35: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_351, permute_197);  view_351 = permute_197 = None
    view_352: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_34, [8, 16, 48, 196]);  bmm_34 = None
    view_353: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_35, [8, 16, 196, 48]);  bmm_35 = None
    permute_198: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_352, [0, 1, 3, 2]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_199: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_89, [0, 2, 3, 1]);  sub_89 = None
    sum_74: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_199, [0, 1, 2], True)
    view_354: "f32[16]" = torch.ops.aten.reshape.default(sum_74, [16]);  sum_74 = None
    clone_176: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_355: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_176, [307328, 16]);  clone_176 = None
    permute_200: "f32[16, 307328]" = torch.ops.aten.permute.default(view_355, [1, 0]);  view_355 = None
    mm_58: "f32[16, 3]" = torch.ops.aten.mm.default(permute_200, view_8);  permute_200 = None
    permute_201: "f32[3, 16]" = torch.ops.aten.permute.default(mm_58, [1, 0]);  mm_58 = None
    permute_202: "f32[16, 3]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    full_default_2: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.full.default([2, 8, 16, 196, 48], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_31: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_198, 0, 1);  permute_198 = None
    select_scatter_32: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_353, 0, 0);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_130: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_31, select_scatter_32);  select_scatter_31 = select_scatter_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_203: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_130, [1, 3, 0, 2, 4]);  add_130 = None
    clone_177: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    view_356: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_177, [8, 196, 1536]);  clone_177 = None
    view_357: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_356, [1568, 1536]);  view_356 = None
    permute_204: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_357, [1, 0])
    mm_59: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_204, view_239);  permute_204 = view_239 = None
    permute_205: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    mm_60: "f32[1568, 768]" = torch.ops.aten.mm.default(view_357, permute_206);  view_357 = permute_206 = None
    view_358: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_60, [8, 196, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_131: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_349, view_358);  view_349 = view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_207: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_202: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_131, primals_48);  primals_48 = None
    mul_203: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_202, 768)
    sum_75: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
    mul_204: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_202, mul_90);  mul_202 = None
    sum_76: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
    mul_205: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_90, sum_76);  sum_76 = None
    sub_92: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_203, sum_75);  mul_203 = sum_75 = None
    sub_93: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_205);  sub_92 = mul_205 = None
    mul_206: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_41, sub_93);  div_41 = sub_93 = None
    mul_207: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_131, mul_90);  mul_90 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_131, [0, 1]);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_127, mul_206);  add_127 = mul_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_359: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_132, [1568, 768])
    mm_61: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_359, permute_208);  permute_208 = None
    permute_209: "f32[768, 1568]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_62: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_209, view_233);  permute_209 = view_233 = None
    permute_210: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_62, [1, 0]);  mm_62 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.reshape.default(sum_79, [768]);  sum_79 = None
    permute_211: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_361: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_61, [8, 196, 3072]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_209: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_210: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, view_232)
    mul_211: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_210, -0.5);  mul_210 = None
    exp_25: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_211);  mul_211 = None
    mul_212: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_213: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_232, mul_212);  view_232 = mul_212 = None
    add_134: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_209, mul_213);  mul_209 = mul_213 = None
    mul_214: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_361, add_134);  view_361 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_362: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_214, [1568, 3072]);  mul_214 = None
    mm_63: "f32[1568, 768]" = torch.ops.aten.mm.default(view_362, permute_212);  permute_212 = None
    permute_213: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_64: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_213, view_231);  permute_213 = view_231 = None
    permute_214: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_64, [1, 0]);  mm_64 = None
    sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[3072]" = torch.ops.aten.reshape.default(sum_80, [3072]);  sum_80 = None
    permute_215: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_364: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_63, [8, 196, 768]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_216: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, primals_46);  primals_46 = None
    mul_217: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_216, 768)
    sum_81: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_216, [2], True)
    mul_218: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_216, mul_85);  mul_216 = None
    sum_82: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True);  mul_218 = None
    mul_219: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_85, sum_82);  sum_82 = None
    sub_95: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_217, sum_81);  mul_217 = sum_81 = None
    sub_96: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_219);  sub_95 = mul_219 = None
    mul_220: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_42, sub_96);  div_42 = sub_96 = None
    mul_221: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_364, mul_85);  mul_85 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 1]);  mul_221 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_364, [0, 1]);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_135: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_132, mul_220);  add_132 = mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_365: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_135, [1568, 768])
    mm_65: "f32[1568, 768]" = torch.ops.aten.mm.default(view_365, permute_216);  permute_216 = None
    permute_217: "f32[768, 1568]" = torch.ops.aten.permute.default(view_365, [1, 0])
    mm_66: "f32[768, 768]" = torch.ops.aten.mm.default(permute_217, view_229);  permute_217 = view_229 = None
    permute_218: "f32[768, 768]" = torch.ops.aten.permute.default(mm_66, [1, 0]);  mm_66 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
    view_366: "f32[768]" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_367: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_65, [8, 196, 768]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_368: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_367, [8, 196, 16, 48]);  view_367 = None
    permute_220: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_180: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
    view_369: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_180, [128, 196, 48]);  clone_180 = None
    bmm_36: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_221, view_369);  permute_221 = None
    bmm_37: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_369, permute_222);  view_369 = permute_222 = None
    view_370: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_36, [8, 16, 196, 48]);  bmm_36 = None
    view_371: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_37, [8, 16, 196, 196]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_223: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    clone_181: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
    view_372: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_181, [8, 196, 768]);  clone_181 = None
    view_373: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_372, [1568, 768]);  view_372 = None
    permute_224: "f32[768, 1568]" = torch.ops.aten.permute.default(view_373, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_224, view_213);  permute_224 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    mm_68: "f32[1568, 768]" = torch.ops.aten.mm.default(view_373, permute_226);  view_373 = permute_226 = None
    view_374: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_68, [8, 196, 768]);  mm_68 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_43: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_85, unsqueeze_53);  add_85 = None
    div_44: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_43, unsqueeze_53);  div_43 = None
    neg_2: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_371)
    mul_222: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_2, div_44);  neg_2 = div_44 = None
    div_45: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_371, unsqueeze_53);  view_371 = unsqueeze_53 = None
    sum_86: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [3], True);  mul_222 = None
    squeeze_1: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_86, -1);  sum_86 = None
    unsqueeze_61: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_1, -1);  squeeze_1 = None
    expand_90: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_61, [8, 16, 196, 196]);  unsqueeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_136: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_45, expand_90);  div_45 = expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_223: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, sigmoid_16)
    mul_224: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, div_25)
    sum_87: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 2, 3], True);  mul_224 = None
    sub_97: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_16)
    mul_225: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_16, sub_97);  sigmoid_16 = sub_97 = None
    mul_226: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_225);  sum_87 = None
    mul_227: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, sub_52);  sub_52 = None
    mul_228: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_136, div_24);  add_136 = None
    sum_88: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [0, 2, 3], True);  mul_228 = None
    neg_3: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_88);  sum_88 = None
    mul_230: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_3, mul_225);  neg_3 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_137: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_226, mul_230);  mul_226 = mul_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_375: "f32[16]" = torch.ops.aten.reshape.default(add_137, [16]);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_231: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_223, div_25);  mul_223 = None
    sum_89: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [-1], True)
    mul_232: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_25, sum_89);  div_25 = sum_89 = None
    sub_99: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_233: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_227, div_24);  mul_227 = None
    sum_90: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_233, [-1], True)
    mul_234: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_24, sum_90);  div_24 = sum_90 = None
    sub_100: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_233, mul_234);  mul_233 = mul_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_235: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_100, 0.14433756729740643);  sub_100 = None
    view_376: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_235, [128, 196, 196]);  mul_235 = None
    bmm_38: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_228, view_376);  permute_228 = None
    bmm_39: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_376, permute_229);  view_376 = permute_229 = None
    view_377: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_38, [8, 16, 48, 196]);  bmm_38 = None
    view_378: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_39, [8, 16, 196, 48]);  bmm_39 = None
    permute_230: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_377, [0, 1, 3, 2]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_231: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_99, [0, 2, 3, 1]);  sub_99 = None
    sum_91: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_231, [0, 1, 2], True)
    view_379: "f32[16]" = torch.ops.aten.reshape.default(sum_91, [16]);  sum_91 = None
    clone_182: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_380: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_182, [307328, 16]);  clone_182 = None
    permute_232: "f32[16, 307328]" = torch.ops.aten.permute.default(view_380, [1, 0]);  view_380 = None
    mm_69: "f32[16, 3]" = torch.ops.aten.mm.default(permute_232, view_8);  permute_232 = None
    permute_233: "f32[3, 16]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    permute_234: "f32[16, 3]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_33: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_230, 0, 1);  permute_230 = None
    select_scatter_34: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_378, 0, 0);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_138: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_33, select_scatter_34);  select_scatter_33 = select_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_235: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_138, [1, 3, 0, 2, 4]);  add_138 = None
    clone_183: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_381: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_183, [8, 196, 1536]);  clone_183 = None
    view_382: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_381, [1568, 1536]);  view_381 = None
    permute_236: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_382, [1, 0])
    mm_70: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_236, view_213);  permute_236 = view_213 = None
    permute_237: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_70, [1, 0]);  mm_70 = None
    mm_71: "f32[1568, 768]" = torch.ops.aten.mm.default(view_382, permute_238);  view_382 = permute_238 = None
    view_383: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_71, [8, 196, 768]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_139: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_374, view_383);  view_374 = view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_239: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_237: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_139, primals_43);  primals_43 = None
    mul_238: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, 768)
    sum_92: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_237, [2], True)
    mul_239: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_237, mul_80);  mul_237 = None
    sum_93: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [2], True);  mul_239 = None
    mul_240: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_80, sum_93);  sum_93 = None
    sub_102: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_238, sum_92);  mul_238 = sum_92 = None
    sub_103: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_102, mul_240);  sub_102 = mul_240 = None
    mul_241: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_46, sub_103);  div_46 = sub_103 = None
    mul_242: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_139, mul_80);  mul_80 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_242, [0, 1]);  mul_242 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_135, mul_241);  add_135 = mul_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_384: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_140, [1568, 768])
    mm_72: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_384, permute_240);  permute_240 = None
    permute_241: "f32[768, 1568]" = torch.ops.aten.permute.default(view_384, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_207);  permute_241 = view_207 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_384, [0], True);  view_384 = None
    view_385: "f32[768]" = torch.ops.aten.reshape.default(sum_96, [768]);  sum_96 = None
    permute_243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_386: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_72, [8, 196, 3072]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_244: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_79, 0.5);  add_79 = None
    mul_245: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, view_206)
    mul_246: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_245, -0.5);  mul_245 = None
    exp_26: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_246);  mul_246 = None
    mul_247: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_248: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_206, mul_247);  view_206 = mul_247 = None
    add_142: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_244, mul_248);  mul_244 = mul_248 = None
    mul_249: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_386, add_142);  view_386 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_387: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_249, [1568, 3072]);  mul_249 = None
    mm_74: "f32[1568, 768]" = torch.ops.aten.mm.default(view_387, permute_244);  permute_244 = None
    permute_245: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_205);  permute_245 = view_205 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_97: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[3072]" = torch.ops.aten.reshape.default(sum_97, [3072]);  sum_97 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_389: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_74, [8, 196, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_251: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_389, primals_41);  primals_41 = None
    mul_252: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
    sum_98: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_75);  mul_251 = None
    sum_99: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_75, sum_99);  sum_99 = None
    sub_105: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_98);  mul_252 = sum_98 = None
    sub_106: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_105, mul_254);  sub_105 = mul_254 = None
    mul_255: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_47, sub_106);  div_47 = sub_106 = None
    mul_256: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_389, mul_75);  mul_75 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_389, [0, 1]);  view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_143: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_140, mul_255);  add_140 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_390: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_143, [1568, 768])
    mm_76: "f32[1568, 768]" = torch.ops.aten.mm.default(view_390, permute_248);  permute_248 = None
    permute_249: "f32[768, 1568]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_249, view_203);  permute_249 = view_203 = None
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[768]" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_392: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_76, [8, 196, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_393: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_392, [8, 196, 16, 48]);  view_392 = None
    permute_252: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_393, [0, 2, 1, 3]);  view_393 = None
    clone_186: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
    view_394: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_186, [128, 196, 48]);  clone_186 = None
    bmm_40: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_253, view_394);  permute_253 = None
    bmm_41: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_394, permute_254);  view_394 = permute_254 = None
    view_395: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_40, [8, 16, 196, 48]);  bmm_40 = None
    view_396: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_41, [8, 16, 196, 196]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_255: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
    clone_187: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    view_397: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_187, [8, 196, 768]);  clone_187 = None
    view_398: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_397, [1568, 768]);  view_397 = None
    permute_256: "f32[768, 1568]" = torch.ops.aten.permute.default(view_398, [1, 0])
    mm_78: "f32[768, 768]" = torch.ops.aten.mm.default(permute_256, view_187);  permute_256 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(mm_78, [1, 0]);  mm_78 = None
    mm_79: "f32[1568, 768]" = torch.ops.aten.mm.default(view_398, permute_258);  view_398 = permute_258 = None
    view_399: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_79, [8, 196, 768]);  mm_79 = None
    permute_259: "f32[768, 768]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_48: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_75, unsqueeze_47);  add_75 = None
    div_49: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_48, unsqueeze_47);  div_48 = None
    neg_4: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_396)
    mul_257: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_4, div_49);  neg_4 = div_49 = None
    div_50: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_396, unsqueeze_47);  view_396 = unsqueeze_47 = None
    sum_103: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [3], True);  mul_257 = None
    squeeze_2: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_103, -1);  sum_103 = None
    unsqueeze_62: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_2, -1);  squeeze_2 = None
    expand_91: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_62, [8, 16, 196, 196]);  unsqueeze_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_144: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_50, expand_91);  div_50 = expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_258: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, sigmoid_14)
    mul_259: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, div_22)
    sum_104: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 2, 3], True);  mul_259 = None
    sub_107: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_14)
    mul_260: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_14, sub_107);  sigmoid_14 = sub_107 = None
    mul_261: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_104, mul_260);  sum_104 = None
    mul_262: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, sub_46);  sub_46 = None
    mul_263: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_144, div_21);  add_144 = None
    sum_105: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3], True);  mul_263 = None
    neg_5: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_105);  sum_105 = None
    mul_265: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_5, mul_260);  neg_5 = mul_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_145: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_261, mul_265);  mul_261 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_400: "f32[16]" = torch.ops.aten.reshape.default(add_145, [16]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_266: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_258, div_22);  mul_258 = None
    sum_106: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [-1], True)
    mul_267: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_22, sum_106);  div_22 = sum_106 = None
    sub_109: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_268: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_262, div_21);  mul_262 = None
    sum_107: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [-1], True)
    mul_269: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_21, sum_107);  div_21 = sum_107 = None
    sub_110: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_270: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_110, 0.14433756729740643);  sub_110 = None
    view_401: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_270, [128, 196, 196]);  mul_270 = None
    bmm_42: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_260, view_401);  permute_260 = None
    bmm_43: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_401, permute_261);  view_401 = permute_261 = None
    view_402: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_42, [8, 16, 48, 196]);  bmm_42 = None
    view_403: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_43, [8, 16, 196, 48]);  bmm_43 = None
    permute_262: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_402, [0, 1, 3, 2]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_263: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_109, [0, 2, 3, 1]);  sub_109 = None
    sum_108: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_263, [0, 1, 2], True)
    view_404: "f32[16]" = torch.ops.aten.reshape.default(sum_108, [16]);  sum_108 = None
    clone_188: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
    view_405: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_188, [307328, 16]);  clone_188 = None
    permute_264: "f32[16, 307328]" = torch.ops.aten.permute.default(view_405, [1, 0]);  view_405 = None
    mm_80: "f32[16, 3]" = torch.ops.aten.mm.default(permute_264, view_8);  permute_264 = None
    permute_265: "f32[3, 16]" = torch.ops.aten.permute.default(mm_80, [1, 0]);  mm_80 = None
    permute_266: "f32[16, 3]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_35: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_262, 0, 1);  permute_262 = None
    select_scatter_36: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_403, 0, 0);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_146: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_35, select_scatter_36);  select_scatter_35 = select_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_267: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_146, [1, 3, 0, 2, 4]);  add_146 = None
    clone_189: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
    view_406: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_189, [8, 196, 1536]);  clone_189 = None
    view_407: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_406, [1568, 1536]);  view_406 = None
    permute_268: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_81: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_268, view_187);  permute_268 = view_187 = None
    permute_269: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    mm_82: "f32[1568, 768]" = torch.ops.aten.mm.default(view_407, permute_270);  view_407 = permute_270 = None
    view_408: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_82, [8, 196, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_147: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_399, view_408);  view_399 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_271: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_272: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_147, primals_38);  primals_38 = None
    mul_273: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_272, 768)
    sum_109: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True)
    mul_274: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_272, mul_70);  mul_272 = None
    sum_110: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True);  mul_274 = None
    mul_275: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_110);  sum_110 = None
    sub_112: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_273, sum_109);  mul_273 = sum_109 = None
    sub_113: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_112, mul_275);  sub_112 = mul_275 = None
    mul_276: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_51, sub_113);  div_51 = sub_113 = None
    mul_277: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_147, mul_70);  mul_70 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_277, [0, 1]);  mul_277 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 1]);  add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_148: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_143, mul_276);  add_143 = mul_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_409: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_148, [1568, 768])
    mm_83: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_409, permute_272);  permute_272 = None
    permute_273: "f32[768, 1568]" = torch.ops.aten.permute.default(view_409, [1, 0])
    mm_84: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_273, view_181);  permute_273 = view_181 = None
    permute_274: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_409, [0], True);  view_409 = None
    view_410: "f32[768]" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
    permute_275: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_411: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_83, [8, 196, 3072]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_279: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_280: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, view_180)
    mul_281: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_280, -0.5);  mul_280 = None
    exp_27: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_281);  mul_281 = None
    mul_282: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_283: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_180, mul_282);  view_180 = mul_282 = None
    add_150: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_279, mul_283);  mul_279 = mul_283 = None
    mul_284: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_411, add_150);  view_411 = add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_412: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_284, [1568, 3072]);  mul_284 = None
    mm_85: "f32[1568, 768]" = torch.ops.aten.mm.default(view_412, permute_276);  permute_276 = None
    permute_277: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_412, [1, 0])
    mm_86: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_277, view_179);  permute_277 = view_179 = None
    permute_278: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_86, [1, 0]);  mm_86 = None
    sum_114: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_412, [0], True);  view_412 = None
    view_413: "f32[3072]" = torch.ops.aten.reshape.default(sum_114, [3072]);  sum_114 = None
    permute_279: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_414: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_85, [8, 196, 768]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_286: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_414, primals_36);  primals_36 = None
    mul_287: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_286, 768)
    sum_115: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_286, mul_65);  mul_286 = None
    sum_116: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_116);  sum_116 = None
    sub_115: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_287, sum_115);  mul_287 = sum_115 = None
    sub_116: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_289);  sub_115 = mul_289 = None
    mul_290: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_52, sub_116);  div_52 = sub_116 = None
    mul_291: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_414, mul_65);  mul_65 = None
    sum_117: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_414, [0, 1]);  view_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_151: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_148, mul_290);  add_148 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_415: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_151, [1568, 768])
    mm_87: "f32[1568, 768]" = torch.ops.aten.mm.default(view_415, permute_280);  permute_280 = None
    permute_281: "f32[768, 1568]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_88: "f32[768, 768]" = torch.ops.aten.mm.default(permute_281, view_177);  permute_281 = view_177 = None
    permute_282: "f32[768, 768]" = torch.ops.aten.permute.default(mm_88, [1, 0]);  mm_88 = None
    sum_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[768]" = torch.ops.aten.reshape.default(sum_119, [768]);  sum_119 = None
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_417: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_87, [8, 196, 768]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_418: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_417, [8, 196, 16, 48]);  view_417 = None
    permute_284: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
    clone_192: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None
    view_419: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_192, [128, 196, 48]);  clone_192 = None
    bmm_44: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_285, view_419);  permute_285 = None
    bmm_45: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_419, permute_286);  view_419 = permute_286 = None
    view_420: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_44, [8, 16, 196, 48]);  bmm_44 = None
    view_421: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_45, [8, 16, 196, 196]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_287: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
    clone_193: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
    view_422: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_193, [8, 196, 768]);  clone_193 = None
    view_423: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_422, [1568, 768]);  view_422 = None
    permute_288: "f32[768, 1568]" = torch.ops.aten.permute.default(view_423, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_288, view_161);  permute_288 = None
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    mm_90: "f32[1568, 768]" = torch.ops.aten.mm.default(view_423, permute_290);  view_423 = permute_290 = None
    view_424: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_90, [8, 196, 768]);  mm_90 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_289, [1, 0]);  permute_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_53: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_65, unsqueeze_41);  add_65 = None
    div_54: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_53, unsqueeze_41);  div_53 = None
    neg_6: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_421)
    mul_292: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_6, div_54);  neg_6 = div_54 = None
    div_55: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_421, unsqueeze_41);  view_421 = unsqueeze_41 = None
    sum_120: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [3], True);  mul_292 = None
    squeeze_3: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_120, -1);  sum_120 = None
    unsqueeze_63: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_3, -1);  squeeze_3 = None
    expand_92: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_63, [8, 16, 196, 196]);  unsqueeze_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_152: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_55, expand_92);  div_55 = expand_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_293: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, sigmoid_12)
    mul_294: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, div_19)
    sum_121: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [0, 2, 3], True);  mul_294 = None
    sub_117: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
    mul_295: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_117);  sigmoid_12 = sub_117 = None
    mul_296: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_121, mul_295);  sum_121 = None
    mul_297: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, sub_40);  sub_40 = None
    mul_298: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_152, div_18);  add_152 = None
    sum_122: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3], True);  mul_298 = None
    neg_7: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_122);  sum_122 = None
    mul_300: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_7, mul_295);  neg_7 = mul_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_153: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_296, mul_300);  mul_296 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_425: "f32[16]" = torch.ops.aten.reshape.default(add_153, [16]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_301: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_293, div_19);  mul_293 = None
    sum_123: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
    mul_302: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_19, sum_123);  div_19 = sum_123 = None
    sub_119: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_301, mul_302);  mul_301 = mul_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_303: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_297, div_18);  mul_297 = None
    sum_124: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [-1], True)
    mul_304: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_18, sum_124);  div_18 = sum_124 = None
    sub_120: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_305: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_120, 0.14433756729740643);  sub_120 = None
    view_426: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_305, [128, 196, 196]);  mul_305 = None
    bmm_46: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_292, view_426);  permute_292 = None
    bmm_47: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_426, permute_293);  view_426 = permute_293 = None
    view_427: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_46, [8, 16, 48, 196]);  bmm_46 = None
    view_428: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_47, [8, 16, 196, 48]);  bmm_47 = None
    permute_294: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_427, [0, 1, 3, 2]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_295: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_119, [0, 2, 3, 1]);  sub_119 = None
    sum_125: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_295, [0, 1, 2], True)
    view_429: "f32[16]" = torch.ops.aten.reshape.default(sum_125, [16]);  sum_125 = None
    clone_194: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_295, memory_format = torch.contiguous_format);  permute_295 = None
    view_430: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_194, [307328, 16]);  clone_194 = None
    permute_296: "f32[16, 307328]" = torch.ops.aten.permute.default(view_430, [1, 0]);  view_430 = None
    mm_91: "f32[16, 3]" = torch.ops.aten.mm.default(permute_296, view_8);  permute_296 = None
    permute_297: "f32[3, 16]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    permute_298: "f32[16, 3]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_37: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_294, 0, 1);  permute_294 = None
    select_scatter_38: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_428, 0, 0);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_154: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_37, select_scatter_38);  select_scatter_37 = select_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_299: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_154, [1, 3, 0, 2, 4]);  add_154 = None
    clone_195: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_299, memory_format = torch.contiguous_format);  permute_299 = None
    view_431: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_195, [8, 196, 1536]);  clone_195 = None
    view_432: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_431, [1568, 1536]);  view_431 = None
    permute_300: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_92: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_300, view_161);  permute_300 = view_161 = None
    permute_301: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_92, [1, 0]);  mm_92 = None
    mm_93: "f32[1568, 768]" = torch.ops.aten.mm.default(view_432, permute_302);  view_432 = permute_302 = None
    view_433: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_93, [8, 196, 768]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_155: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_424, view_433);  view_424 = view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_303: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_307: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_155, primals_33);  primals_33 = None
    mul_308: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_307, 768)
    sum_126: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
    mul_309: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_307, mul_60);  mul_307 = None
    sum_127: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
    mul_310: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_60, sum_127);  sum_127 = None
    sub_122: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_308, sum_126);  mul_308 = sum_126 = None
    sub_123: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_310);  sub_122 = mul_310 = None
    mul_311: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_123);  div_56 = sub_123 = None
    mul_312: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_155, mul_60);  mul_60 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_155, [0, 1]);  add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_156: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_151, mul_311);  add_151 = mul_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_156, [1568, 768])
    mm_94: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_434, permute_304);  permute_304 = None
    permute_305: "f32[768, 1568]" = torch.ops.aten.permute.default(view_434, [1, 0])
    mm_95: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_305, view_155);  permute_305 = view_155 = None
    permute_306: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
    view_435: "f32[768]" = torch.ops.aten.reshape.default(sum_130, [768]);  sum_130 = None
    permute_307: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_436: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_94, [8, 196, 3072]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_314: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_59, 0.5);  add_59 = None
    mul_315: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, view_154)
    mul_316: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_315, -0.5);  mul_315 = None
    exp_28: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_316);  mul_316 = None
    mul_317: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_318: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_154, mul_317);  view_154 = mul_317 = None
    add_158: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_314, mul_318);  mul_314 = mul_318 = None
    mul_319: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_436, add_158);  view_436 = add_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_437: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_319, [1568, 3072]);  mul_319 = None
    mm_96: "f32[1568, 768]" = torch.ops.aten.mm.default(view_437, permute_308);  permute_308 = None
    permute_309: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_437, [1, 0])
    mm_97: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_309, view_153);  permute_309 = view_153 = None
    permute_310: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_131: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
    view_438: "f32[3072]" = torch.ops.aten.reshape.default(sum_131, [3072]);  sum_131 = None
    permute_311: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_439: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_96, [8, 196, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_321: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_439, primals_31);  primals_31 = None
    mul_322: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, 768)
    sum_132: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True)
    mul_323: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_321, mul_55);  mul_321 = None
    sum_133: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True);  mul_323 = None
    mul_324: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_55, sum_133);  sum_133 = None
    sub_125: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_322, sum_132);  mul_322 = sum_132 = None
    sub_126: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_324);  sub_125 = mul_324 = None
    mul_325: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_57, sub_126);  div_57 = sub_126 = None
    mul_326: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_439, mul_55);  mul_55 = None
    sum_134: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_326, [0, 1]);  mul_326 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_439, [0, 1]);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_159: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_156, mul_325);  add_156 = mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_440: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_159, [1568, 768])
    mm_98: "f32[1568, 768]" = torch.ops.aten.mm.default(view_440, permute_312);  permute_312 = None
    permute_313: "f32[768, 1568]" = torch.ops.aten.permute.default(view_440, [1, 0])
    mm_99: "f32[768, 768]" = torch.ops.aten.mm.default(permute_313, view_151);  permute_313 = view_151 = None
    permute_314: "f32[768, 768]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_136: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
    view_441: "f32[768]" = torch.ops.aten.reshape.default(sum_136, [768]);  sum_136 = None
    permute_315: "f32[768, 768]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    view_442: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_98, [8, 196, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_443: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_442, [8, 196, 16, 48]);  view_442 = None
    permute_316: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    clone_198: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_444: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_198, [128, 196, 48]);  clone_198 = None
    bmm_48: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_317, view_444);  permute_317 = None
    bmm_49: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_444, permute_318);  view_444 = permute_318 = None
    view_445: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_48, [8, 16, 196, 48]);  bmm_48 = None
    view_446: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_49, [8, 16, 196, 196]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_319: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    clone_199: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None
    view_447: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_199, [8, 196, 768]);  clone_199 = None
    view_448: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_447, [1568, 768]);  view_447 = None
    permute_320: "f32[768, 1568]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_100: "f32[768, 768]" = torch.ops.aten.mm.default(permute_320, view_135);  permute_320 = None
    permute_321: "f32[768, 768]" = torch.ops.aten.permute.default(mm_100, [1, 0]);  mm_100 = None
    mm_101: "f32[1568, 768]" = torch.ops.aten.mm.default(view_448, permute_322);  view_448 = permute_322 = None
    view_449: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_101, [8, 196, 768]);  mm_101 = None
    permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_58: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_55, unsqueeze_35);  add_55 = None
    div_59: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_58, unsqueeze_35);  div_58 = None
    neg_8: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_446)
    mul_327: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_8, div_59);  neg_8 = div_59 = None
    div_60: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_446, unsqueeze_35);  view_446 = unsqueeze_35 = None
    sum_137: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [3], True);  mul_327 = None
    squeeze_4: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_137, -1);  sum_137 = None
    unsqueeze_64: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_4, -1);  squeeze_4 = None
    expand_93: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_64, [8, 16, 196, 196]);  unsqueeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_160: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_60, expand_93);  div_60 = expand_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_328: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, sigmoid_10)
    mul_329: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, div_16)
    sum_138: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 2, 3], True);  mul_329 = None
    sub_127: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
    mul_330: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_127);  sigmoid_10 = sub_127 = None
    mul_331: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_138, mul_330);  sum_138 = None
    mul_332: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, sub_34);  sub_34 = None
    mul_333: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_160, div_15);  add_160 = None
    sum_139: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3], True);  mul_333 = None
    neg_9: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_139);  sum_139 = None
    mul_335: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_9, mul_330);  neg_9 = mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_161: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_331, mul_335);  mul_331 = mul_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_450: "f32[16]" = torch.ops.aten.reshape.default(add_161, [16]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_336: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_328, div_16);  mul_328 = None
    sum_140: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [-1], True)
    mul_337: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_16, sum_140);  div_16 = sum_140 = None
    sub_129: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_338: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_332, div_15);  mul_332 = None
    sum_141: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [-1], True)
    mul_339: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_15, sum_141);  div_15 = sum_141 = None
    sub_130: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_340: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_130, 0.14433756729740643);  sub_130 = None
    view_451: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_340, [128, 196, 196]);  mul_340 = None
    bmm_50: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_324, view_451);  permute_324 = None
    bmm_51: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_451, permute_325);  view_451 = permute_325 = None
    view_452: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_50, [8, 16, 48, 196]);  bmm_50 = None
    view_453: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_51, [8, 16, 196, 48]);  bmm_51 = None
    permute_326: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_452, [0, 1, 3, 2]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_327: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_129, [0, 2, 3, 1]);  sub_129 = None
    sum_142: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_327, [0, 1, 2], True)
    view_454: "f32[16]" = torch.ops.aten.reshape.default(sum_142, [16]);  sum_142 = None
    clone_200: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
    view_455: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_200, [307328, 16]);  clone_200 = None
    permute_328: "f32[16, 307328]" = torch.ops.aten.permute.default(view_455, [1, 0]);  view_455 = None
    mm_102: "f32[16, 3]" = torch.ops.aten.mm.default(permute_328, view_8);  permute_328 = None
    permute_329: "f32[3, 16]" = torch.ops.aten.permute.default(mm_102, [1, 0]);  mm_102 = None
    permute_330: "f32[16, 3]" = torch.ops.aten.permute.default(permute_329, [1, 0]);  permute_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_39: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_326, 0, 1);  permute_326 = None
    select_scatter_40: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_453, 0, 0);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_162: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_39, select_scatter_40);  select_scatter_39 = select_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_331: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_162, [1, 3, 0, 2, 4]);  add_162 = None
    clone_201: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_456: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_201, [8, 196, 1536]);  clone_201 = None
    view_457: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_456, [1568, 1536]);  view_456 = None
    permute_332: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_457, [1, 0])
    mm_103: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_332, view_135);  permute_332 = view_135 = None
    permute_333: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    mm_104: "f32[1568, 768]" = torch.ops.aten.mm.default(view_457, permute_334);  view_457 = permute_334 = None
    view_458: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_104, [8, 196, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_163: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_449, view_458);  view_449 = view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_335: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_342: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_163, primals_28);  primals_28 = None
    mul_343: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_342, 768)
    sum_143: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
    mul_344: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_342, mul_50);  mul_342 = None
    sum_144: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
    mul_345: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_50, sum_144);  sum_144 = None
    sub_132: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_343, sum_143);  mul_343 = sum_143 = None
    sub_133: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_345);  sub_132 = mul_345 = None
    mul_346: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_133);  div_61 = sub_133 = None
    mul_347: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_163, mul_50);  mul_50 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
    sum_146: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_164: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_159, mul_346);  add_159 = mul_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_164, [1568, 768])
    mm_105: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_459, permute_336);  permute_336 = None
    permute_337: "f32[768, 1568]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_106: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_337, view_129);  permute_337 = view_129 = None
    permute_338: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_106, [1, 0]);  mm_106 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
    permute_339: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
    view_461: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_105, [8, 196, 3072]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_349: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_350: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, view_128)
    mul_351: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_350, -0.5);  mul_350 = None
    exp_29: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_351);  mul_351 = None
    mul_352: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_353: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_128, mul_352);  view_128 = mul_352 = None
    add_166: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_349, mul_353);  mul_349 = mul_353 = None
    mul_354: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_461, add_166);  view_461 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_354, [1568, 3072]);  mul_354 = None
    mm_107: "f32[1568, 768]" = torch.ops.aten.mm.default(view_462, permute_340);  permute_340 = None
    permute_341: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_108: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_341, view_127);  permute_341 = view_127 = None
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_108, [1, 0]);  mm_108 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[3072]" = torch.ops.aten.reshape.default(sum_148, [3072]);  sum_148 = None
    permute_343: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
    view_464: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_107, [8, 196, 768]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_356: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_464, primals_26);  primals_26 = None
    mul_357: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_356, 768)
    sum_149: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True)
    mul_358: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_356, mul_45);  mul_356 = None
    sum_150: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True);  mul_358 = None
    mul_359: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_45, sum_150);  sum_150 = None
    sub_135: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_357, sum_149);  mul_357 = sum_149 = None
    sub_136: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_359);  sub_135 = mul_359 = None
    mul_360: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_62, sub_136);  div_62 = sub_136 = None
    mul_361: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_464, mul_45);  mul_45 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1]);  mul_361 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_464, [0, 1]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_167: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_164, mul_360);  add_164 = mul_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_465: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_167, [1568, 768])
    mm_109: "f32[1568, 768]" = torch.ops.aten.mm.default(view_465, permute_344);  permute_344 = None
    permute_345: "f32[768, 1568]" = torch.ops.aten.permute.default(view_465, [1, 0])
    mm_110: "f32[768, 768]" = torch.ops.aten.mm.default(permute_345, view_125);  permute_345 = view_125 = None
    permute_346: "f32[768, 768]" = torch.ops.aten.permute.default(mm_110, [1, 0]);  mm_110 = None
    sum_153: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
    view_466: "f32[768]" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
    permute_347: "f32[768, 768]" = torch.ops.aten.permute.default(permute_346, [1, 0]);  permute_346 = None
    view_467: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_109, [8, 196, 768]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_468: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_467, [8, 196, 16, 48]);  view_467 = None
    permute_348: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_204: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_469: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_204, [128, 196, 48]);  clone_204 = None
    bmm_52: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_349, view_469);  permute_349 = None
    bmm_53: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_469, permute_350);  view_469 = permute_350 = None
    view_470: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_52, [8, 16, 196, 48]);  bmm_52 = None
    view_471: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_53, [8, 16, 196, 196]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_351: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    clone_205: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_351, memory_format = torch.contiguous_format);  permute_351 = None
    view_472: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_205, [8, 196, 768]);  clone_205 = None
    view_473: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_472, [1568, 768]);  view_472 = None
    permute_352: "f32[768, 1568]" = torch.ops.aten.permute.default(view_473, [1, 0])
    mm_111: "f32[768, 768]" = torch.ops.aten.mm.default(permute_352, view_109);  permute_352 = None
    permute_353: "f32[768, 768]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    mm_112: "f32[1568, 768]" = torch.ops.aten.mm.default(view_473, permute_354);  view_473 = permute_354 = None
    view_474: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_112, [8, 196, 768]);  mm_112 = None
    permute_355: "f32[768, 768]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_63: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_45, unsqueeze_29);  add_45 = None
    div_64: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_63, unsqueeze_29);  div_63 = None
    neg_10: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_471)
    mul_362: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_10, div_64);  neg_10 = div_64 = None
    div_65: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_471, unsqueeze_29);  view_471 = unsqueeze_29 = None
    sum_154: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [3], True);  mul_362 = None
    squeeze_5: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_154, -1);  sum_154 = None
    unsqueeze_65: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_5, -1);  squeeze_5 = None
    expand_94: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_65, [8, 16, 196, 196]);  unsqueeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_168: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_65, expand_94);  div_65 = expand_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_363: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, sigmoid_8)
    mul_364: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, div_13)
    sum_155: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3], True);  mul_364 = None
    sub_137: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
    mul_365: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_137);  sigmoid_8 = sub_137 = None
    mul_366: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_155, mul_365);  sum_155 = None
    mul_367: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, sub_28);  sub_28 = None
    mul_368: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_168, div_12);  add_168 = None
    sum_156: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3], True);  mul_368 = None
    neg_11: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_156);  sum_156 = None
    mul_370: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_11, mul_365);  neg_11 = mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_169: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_366, mul_370);  mul_366 = mul_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_475: "f32[16]" = torch.ops.aten.reshape.default(add_169, [16]);  add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_371: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_363, div_13);  mul_363 = None
    sum_157: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [-1], True)
    mul_372: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_13, sum_157);  div_13 = sum_157 = None
    sub_139: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_371, mul_372);  mul_371 = mul_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_373: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_367, div_12);  mul_367 = None
    sum_158: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [-1], True)
    mul_374: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_12, sum_158);  div_12 = sum_158 = None
    sub_140: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_373, mul_374);  mul_373 = mul_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_375: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_140, 0.14433756729740643);  sub_140 = None
    view_476: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_375, [128, 196, 196]);  mul_375 = None
    bmm_54: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_356, view_476);  permute_356 = None
    bmm_55: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_476, permute_357);  view_476 = permute_357 = None
    view_477: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_54, [8, 16, 48, 196]);  bmm_54 = None
    view_478: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_55, [8, 16, 196, 48]);  bmm_55 = None
    permute_358: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_477, [0, 1, 3, 2]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_359: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_139, [0, 2, 3, 1]);  sub_139 = None
    sum_159: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_359, [0, 1, 2], True)
    view_479: "f32[16]" = torch.ops.aten.reshape.default(sum_159, [16]);  sum_159 = None
    clone_206: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_359, memory_format = torch.contiguous_format);  permute_359 = None
    view_480: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_206, [307328, 16]);  clone_206 = None
    permute_360: "f32[16, 307328]" = torch.ops.aten.permute.default(view_480, [1, 0]);  view_480 = None
    mm_113: "f32[16, 3]" = torch.ops.aten.mm.default(permute_360, view_8);  permute_360 = None
    permute_361: "f32[3, 16]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    permute_362: "f32[16, 3]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_41: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_358, 0, 1);  permute_358 = None
    select_scatter_42: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_478, 0, 0);  view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_170: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_41, select_scatter_42);  select_scatter_41 = select_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_363: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_170, [1, 3, 0, 2, 4]);  add_170 = None
    clone_207: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_363, memory_format = torch.contiguous_format);  permute_363 = None
    view_481: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_207, [8, 196, 1536]);  clone_207 = None
    view_482: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_481, [1568, 1536]);  view_481 = None
    permute_364: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_482, [1, 0])
    mm_114: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_364, view_109);  permute_364 = view_109 = None
    permute_365: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_114, [1, 0]);  mm_114 = None
    mm_115: "f32[1568, 768]" = torch.ops.aten.mm.default(view_482, permute_366);  view_482 = permute_366 = None
    view_483: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_115, [8, 196, 768]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_171: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_474, view_483);  view_474 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_367: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_377: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_171, primals_23);  primals_23 = None
    mul_378: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_377, 768)
    sum_160: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
    mul_379: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_377, mul_40);  mul_377 = None
    sum_161: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
    mul_380: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_40, sum_161);  sum_161 = None
    sub_142: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_378, sum_160);  mul_378 = sum_160 = None
    sub_143: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_142, mul_380);  sub_142 = mul_380 = None
    mul_381: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_66, sub_143);  div_66 = sub_143 = None
    mul_382: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_171, mul_40);  mul_40 = None
    sum_162: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
    sum_163: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_171, [0, 1]);  add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_172: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_167, mul_381);  add_167 = mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_484: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_172, [1568, 768])
    mm_116: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_484, permute_368);  permute_368 = None
    permute_369: "f32[768, 1568]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_117: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_369, view_103);  permute_369 = view_103 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_164: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[768]" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
    permute_371: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_486: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_116, [8, 196, 3072]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_384: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_39, 0.5);  add_39 = None
    mul_385: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, view_102)
    mul_386: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_385, -0.5);  mul_385 = None
    exp_30: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_386);  mul_386 = None
    mul_387: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
    mul_388: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_102, mul_387);  view_102 = mul_387 = None
    add_174: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_384, mul_388);  mul_384 = mul_388 = None
    mul_389: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_486, add_174);  view_486 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_487: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_389, [1568, 3072]);  mul_389 = None
    mm_118: "f32[1568, 768]" = torch.ops.aten.mm.default(view_487, permute_372);  permute_372 = None
    permute_373: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_487, [1, 0])
    mm_119: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_373, view_101);  permute_373 = view_101 = None
    permute_374: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_165: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
    view_488: "f32[3072]" = torch.ops.aten.reshape.default(sum_165, [3072]);  sum_165 = None
    permute_375: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_489: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_118, [8, 196, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_391: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_489, primals_21);  primals_21 = None
    mul_392: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_391, 768)
    sum_166: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [2], True)
    mul_393: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_35);  mul_391 = None
    sum_167: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True);  mul_393 = None
    mul_394: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_35, sum_167);  sum_167 = None
    sub_145: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_392, sum_166);  mul_392 = sum_166 = None
    sub_146: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_145, mul_394);  sub_145 = mul_394 = None
    mul_395: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_146);  div_67 = sub_146 = None
    mul_396: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_489, mul_35);  mul_35 = None
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 1]);  mul_396 = None
    sum_169: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_489, [0, 1]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_175: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_172, mul_395);  add_172 = mul_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_490: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_175, [1568, 768])
    mm_120: "f32[1568, 768]" = torch.ops.aten.mm.default(view_490, permute_376);  permute_376 = None
    permute_377: "f32[768, 1568]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_121: "f32[768, 768]" = torch.ops.aten.mm.default(permute_377, view_99);  permute_377 = view_99 = None
    permute_378: "f32[768, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
    permute_379: "f32[768, 768]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_492: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_120, [8, 196, 768]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_493: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_492, [8, 196, 16, 48]);  view_492 = None
    permute_380: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    clone_210: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
    view_494: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_210, [128, 196, 48]);  clone_210 = None
    bmm_56: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_381, view_494);  permute_381 = None
    bmm_57: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_494, permute_382);  view_494 = permute_382 = None
    view_495: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_56, [8, 16, 196, 48]);  bmm_56 = None
    view_496: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_57, [8, 16, 196, 196]);  bmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_383: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None
    clone_211: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_383, memory_format = torch.contiguous_format);  permute_383 = None
    view_497: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_211, [8, 196, 768]);  clone_211 = None
    view_498: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_497, [1568, 768]);  view_497 = None
    permute_384: "f32[768, 1568]" = torch.ops.aten.permute.default(view_498, [1, 0])
    mm_122: "f32[768, 768]" = torch.ops.aten.mm.default(permute_384, view_83);  permute_384 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(mm_122, [1, 0]);  mm_122 = None
    mm_123: "f32[1568, 768]" = torch.ops.aten.mm.default(view_498, permute_386);  view_498 = permute_386 = None
    view_499: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_123, [8, 196, 768]);  mm_123 = None
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_68: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_35, unsqueeze_23);  add_35 = None
    div_69: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_68, unsqueeze_23);  div_68 = None
    neg_12: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_496)
    mul_397: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_12, div_69);  neg_12 = div_69 = None
    div_70: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_496, unsqueeze_23);  view_496 = unsqueeze_23 = None
    sum_171: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [3], True);  mul_397 = None
    squeeze_6: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_171, -1);  sum_171 = None
    unsqueeze_66: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_6, -1);  squeeze_6 = None
    expand_95: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_66, [8, 16, 196, 196]);  unsqueeze_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_176: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_70, expand_95);  div_70 = expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_398: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, sigmoid_6)
    mul_399: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, div_10)
    sum_172: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3], True);  mul_399 = None
    sub_147: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
    mul_400: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_147);  sigmoid_6 = sub_147 = None
    mul_401: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_172, mul_400);  sum_172 = None
    mul_402: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, sub_22);  sub_22 = None
    mul_403: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_176, div_9);  add_176 = None
    sum_173: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 2, 3], True);  mul_403 = None
    neg_13: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_173);  sum_173 = None
    mul_405: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_13, mul_400);  neg_13 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_177: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_401, mul_405);  mul_401 = mul_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_500: "f32[16]" = torch.ops.aten.reshape.default(add_177, [16]);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_406: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_398, div_10);  mul_398 = None
    sum_174: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_406, [-1], True)
    mul_407: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_10, sum_174);  div_10 = sum_174 = None
    sub_149: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_408: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_402, div_9);  mul_402 = None
    sum_175: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_408, [-1], True)
    mul_409: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_9, sum_175);  div_9 = sum_175 = None
    sub_150: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_410: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_150, 0.14433756729740643);  sub_150 = None
    view_501: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_410, [128, 196, 196]);  mul_410 = None
    bmm_58: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_388, view_501);  permute_388 = None
    bmm_59: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_501, permute_389);  view_501 = permute_389 = None
    view_502: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_58, [8, 16, 48, 196]);  bmm_58 = None
    view_503: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_59, [8, 16, 196, 48]);  bmm_59 = None
    permute_390: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_502, [0, 1, 3, 2]);  view_502 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_391: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_149, [0, 2, 3, 1]);  sub_149 = None
    sum_176: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_391, [0, 1, 2], True)
    view_504: "f32[16]" = torch.ops.aten.reshape.default(sum_176, [16]);  sum_176 = None
    clone_212: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_505: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_212, [307328, 16]);  clone_212 = None
    permute_392: "f32[16, 307328]" = torch.ops.aten.permute.default(view_505, [1, 0]);  view_505 = None
    mm_124: "f32[16, 3]" = torch.ops.aten.mm.default(permute_392, view_8);  permute_392 = None
    permute_393: "f32[3, 16]" = torch.ops.aten.permute.default(mm_124, [1, 0]);  mm_124 = None
    permute_394: "f32[16, 3]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_43: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_390, 0, 1);  permute_390 = None
    select_scatter_44: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_503, 0, 0);  view_503 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_178: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_43, select_scatter_44);  select_scatter_43 = select_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_395: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_178, [1, 3, 0, 2, 4]);  add_178 = None
    clone_213: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
    view_506: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_213, [8, 196, 1536]);  clone_213 = None
    view_507: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_506, [1568, 1536]);  view_506 = None
    permute_396: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_125: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_396, view_83);  permute_396 = view_83 = None
    permute_397: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    mm_126: "f32[1568, 768]" = torch.ops.aten.mm.default(view_507, permute_398);  view_507 = permute_398 = None
    view_508: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_126, [8, 196, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_179: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_499, view_508);  view_499 = view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_399: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_412: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_179, primals_18);  primals_18 = None
    mul_413: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
    sum_177: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_30);  mul_412 = None
    sum_178: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_178);  sum_178 = None
    sub_152: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_177);  mul_413 = sum_177 = None
    sub_153: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_152, mul_415);  sub_152 = mul_415 = None
    mul_416: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_71, sub_153);  div_71 = sub_153 = None
    mul_417: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_179, mul_30);  mul_30 = None
    sum_179: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_180: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_175, mul_416);  add_175 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_509: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_180, [1568, 768])
    mm_127: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_509, permute_400);  permute_400 = None
    permute_401: "f32[768, 1568]" = torch.ops.aten.permute.default(view_509, [1, 0])
    mm_128: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_401, view_77);  permute_401 = view_77 = None
    permute_402: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_128, [1, 0]);  mm_128 = None
    sum_181: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
    view_510: "f32[768]" = torch.ops.aten.reshape.default(sum_181, [768]);  sum_181 = None
    permute_403: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_511: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_127, [8, 196, 3072]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_419: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
    mul_420: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, view_76)
    mul_421: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_420, -0.5);  mul_420 = None
    exp_31: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_421);  mul_421 = None
    mul_422: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
    mul_423: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_76, mul_422);  view_76 = mul_422 = None
    add_182: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_419, mul_423);  mul_419 = mul_423 = None
    mul_424: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_511, add_182);  view_511 = add_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_512: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_424, [1568, 3072]);  mul_424 = None
    mm_129: "f32[1568, 768]" = torch.ops.aten.mm.default(view_512, permute_404);  permute_404 = None
    permute_405: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_512, [1, 0])
    mm_130: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_405, view_75);  permute_405 = view_75 = None
    permute_406: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    sum_182: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True);  view_512 = None
    view_513: "f32[3072]" = torch.ops.aten.reshape.default(sum_182, [3072]);  sum_182 = None
    permute_407: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_514: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_129, [8, 196, 768]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_426: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_514, primals_16);  primals_16 = None
    mul_427: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_426, 768)
    sum_183: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [2], True)
    mul_428: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_426, mul_25);  mul_426 = None
    sum_184: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True);  mul_428 = None
    mul_429: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_184);  sum_184 = None
    sub_155: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_427, sum_183);  mul_427 = sum_183 = None
    sub_156: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_155, mul_429);  sub_155 = mul_429 = None
    mul_430: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_156);  div_72 = sub_156 = None
    mul_431: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_514, mul_25);  mul_25 = None
    sum_185: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 1]);  mul_431 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_514, [0, 1]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_183: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_180, mul_430);  add_180 = mul_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_515: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_183, [1568, 768])
    mm_131: "f32[1568, 768]" = torch.ops.aten.mm.default(view_515, permute_408);  permute_408 = None
    permute_409: "f32[768, 1568]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_132: "f32[768, 768]" = torch.ops.aten.mm.default(permute_409, view_73);  permute_409 = view_73 = None
    permute_410: "f32[768, 768]" = torch.ops.aten.permute.default(mm_132, [1, 0]);  mm_132 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.reshape.default(sum_187, [768]);  sum_187 = None
    permute_411: "f32[768, 768]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_517: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_131, [8, 196, 768]);  mm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_518: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_517, [8, 196, 16, 48]);  view_517 = None
    permute_412: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
    clone_216: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_412, memory_format = torch.contiguous_format);  permute_412 = None
    view_519: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_216, [128, 196, 48]);  clone_216 = None
    bmm_60: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_413, view_519);  permute_413 = None
    bmm_61: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_519, permute_414);  view_519 = permute_414 = None
    view_520: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_60, [8, 16, 196, 48]);  bmm_60 = None
    view_521: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_61, [8, 16, 196, 196]);  bmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_415: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
    clone_217: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_415, memory_format = torch.contiguous_format);  permute_415 = None
    view_522: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_217, [8, 196, 768]);  clone_217 = None
    view_523: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_522, [1568, 768]);  view_522 = None
    permute_416: "f32[768, 1568]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_133: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_57);  permute_416 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    mm_134: "f32[1568, 768]" = torch.ops.aten.mm.default(view_523, permute_418);  view_523 = permute_418 = None
    view_524: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_134, [8, 196, 768]);  mm_134 = None
    permute_419: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_73: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_25, unsqueeze_17);  add_25 = None
    div_74: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_73, unsqueeze_17);  div_73 = None
    neg_14: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_521)
    mul_432: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_14, div_74);  neg_14 = div_74 = None
    div_75: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_521, unsqueeze_17);  view_521 = unsqueeze_17 = None
    sum_188: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [3], True);  mul_432 = None
    squeeze_7: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_188, -1);  sum_188 = None
    unsqueeze_67: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_7, -1);  squeeze_7 = None
    expand_96: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_67, [8, 16, 196, 196]);  unsqueeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_184: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_75, expand_96);  div_75 = expand_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_433: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, sigmoid_4)
    mul_434: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, div_7)
    sum_189: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [0, 2, 3], True);  mul_434 = None
    sub_157: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
    mul_435: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_157);  sigmoid_4 = sub_157 = None
    mul_436: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_189, mul_435);  sum_189 = None
    mul_437: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, sub_16);  sub_16 = None
    mul_438: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_184, div_6);  add_184 = None
    sum_190: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3], True);  mul_438 = None
    neg_15: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_190);  sum_190 = None
    mul_440: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_15, mul_435);  neg_15 = mul_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_185: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_436, mul_440);  mul_436 = mul_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_525: "f32[16]" = torch.ops.aten.reshape.default(add_185, [16]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_441: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_433, div_7);  mul_433 = None
    sum_191: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [-1], True)
    mul_442: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_7, sum_191);  div_7 = sum_191 = None
    sub_159: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_443: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_437, div_6);  mul_437 = None
    sum_192: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [-1], True)
    mul_444: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_6, sum_192);  div_6 = sum_192 = None
    sub_160: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_445: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_160, 0.14433756729740643);  sub_160 = None
    view_526: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_445, [128, 196, 196]);  mul_445 = None
    bmm_62: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_420, view_526);  permute_420 = None
    bmm_63: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_526, permute_421);  view_526 = permute_421 = None
    view_527: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_62, [8, 16, 48, 196]);  bmm_62 = None
    view_528: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_63, [8, 16, 196, 48]);  bmm_63 = None
    permute_422: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_527, [0, 1, 3, 2]);  view_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_423: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_159, [0, 2, 3, 1]);  sub_159 = None
    sum_193: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_423, [0, 1, 2], True)
    view_529: "f32[16]" = torch.ops.aten.reshape.default(sum_193, [16]);  sum_193 = None
    clone_218: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_423, memory_format = torch.contiguous_format);  permute_423 = None
    view_530: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_218, [307328, 16]);  clone_218 = None
    permute_424: "f32[16, 307328]" = torch.ops.aten.permute.default(view_530, [1, 0]);  view_530 = None
    mm_135: "f32[16, 3]" = torch.ops.aten.mm.default(permute_424, view_8);  permute_424 = None
    permute_425: "f32[3, 16]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    permute_426: "f32[16, 3]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_45: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_422, 0, 1);  permute_422 = None
    select_scatter_46: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_528, 0, 0);  view_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_186: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_45, select_scatter_46);  select_scatter_45 = select_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_427: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_186, [1, 3, 0, 2, 4]);  add_186 = None
    clone_219: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
    view_531: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_219, [8, 196, 1536]);  clone_219 = None
    view_532: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_531, [1568, 1536]);  view_531 = None
    permute_428: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_136: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_428, view_57);  permute_428 = view_57 = None
    permute_429: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_136, [1, 0]);  mm_136 = None
    mm_137: "f32[1568, 768]" = torch.ops.aten.mm.default(view_532, permute_430);  view_532 = permute_430 = None
    view_533: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_137, [8, 196, 768]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_187: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_524, view_533);  view_524 = view_533 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_431: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_429, [1, 0]);  permute_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_447: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_187, primals_13);  primals_13 = None
    mul_448: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_447, 768)
    sum_194: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
    mul_449: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_447, mul_20);  mul_447 = None
    sum_195: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
    mul_450: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_20, sum_195);  sum_195 = None
    sub_162: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_448, sum_194);  mul_448 = sum_194 = None
    sub_163: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_162, mul_450);  sub_162 = mul_450 = None
    mul_451: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_76, sub_163);  div_76 = sub_163 = None
    mul_452: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_187, mul_20);  mul_20 = None
    sum_196: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    sum_197: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_187, [0, 1]);  add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_188: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_183, mul_451);  add_183 = mul_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_534: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_188, [1568, 768])
    mm_138: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_534, permute_432);  permute_432 = None
    permute_433: "f32[768, 1568]" = torch.ops.aten.permute.default(view_534, [1, 0])
    mm_139: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_433, view_51);  permute_433 = view_51 = None
    permute_434: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_198: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
    view_535: "f32[768]" = torch.ops.aten.reshape.default(sum_198, [768]);  sum_198 = None
    permute_435: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
    view_536: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_138, [8, 196, 3072]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_454: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_19, 0.5);  add_19 = None
    mul_455: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, view_50)
    mul_456: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_455, -0.5);  mul_455 = None
    exp_32: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_456);  mul_456 = None
    mul_457: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
    mul_458: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_50, mul_457);  view_50 = mul_457 = None
    add_190: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_454, mul_458);  mul_454 = mul_458 = None
    mul_459: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_536, add_190);  view_536 = add_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_537: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_459, [1568, 3072]);  mul_459 = None
    mm_140: "f32[1568, 768]" = torch.ops.aten.mm.default(view_537, permute_436);  permute_436 = None
    permute_437: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_537, [1, 0])
    mm_141: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_437, view_49);  permute_437 = view_49 = None
    permute_438: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_199: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_537, [0], True);  view_537 = None
    view_538: "f32[3072]" = torch.ops.aten.reshape.default(sum_199, [3072]);  sum_199 = None
    permute_439: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
    view_539: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_140, [8, 196, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_461: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_539, primals_11);  primals_11 = None
    mul_462: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_461, 768)
    sum_200: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True)
    mul_463: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_461, mul_15);  mul_461 = None
    sum_201: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_463, [2], True);  mul_463 = None
    mul_464: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_15, sum_201);  sum_201 = None
    sub_165: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_462, sum_200);  mul_462 = sum_200 = None
    sub_166: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_165, mul_464);  sub_165 = mul_464 = None
    mul_465: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_77, sub_166);  div_77 = sub_166 = None
    mul_466: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_539, mul_15);  mul_15 = None
    sum_202: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 1]);  mul_466 = None
    sum_203: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_539, [0, 1]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_191: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_188, mul_465);  add_188 = mul_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_540: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_191, [1568, 768])
    mm_142: "f32[1568, 768]" = torch.ops.aten.mm.default(view_540, permute_440);  permute_440 = None
    permute_441: "f32[768, 1568]" = torch.ops.aten.permute.default(view_540, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_441, view_47);  permute_441 = view_47 = None
    permute_442: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_204: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True);  view_540 = None
    view_541: "f32[768]" = torch.ops.aten.reshape.default(sum_204, [768]);  sum_204 = None
    permute_443: "f32[768, 768]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
    view_542: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_142, [8, 196, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_543: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_542, [8, 196, 16, 48]);  view_542 = None
    permute_444: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_543, [0, 2, 1, 3]);  view_543 = None
    clone_222: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_444, memory_format = torch.contiguous_format);  permute_444 = None
    view_544: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_222, [128, 196, 48]);  clone_222 = None
    bmm_64: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_445, view_544);  permute_445 = None
    bmm_65: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_544, permute_446);  view_544 = permute_446 = None
    view_545: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_64, [8, 16, 196, 48]);  bmm_64 = None
    view_546: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_65, [8, 16, 196, 196]);  bmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_447: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None
    clone_223: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_547: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_223, [8, 196, 768]);  clone_223 = None
    view_548: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_547, [1568, 768]);  view_547 = None
    permute_448: "f32[768, 1568]" = torch.ops.aten.permute.default(view_548, [1, 0])
    mm_144: "f32[768, 768]" = torch.ops.aten.mm.default(permute_448, view_31);  permute_448 = None
    permute_449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_144, [1, 0]);  mm_144 = None
    mm_145: "f32[1568, 768]" = torch.ops.aten.mm.default(view_548, permute_450);  view_548 = permute_450 = None
    view_549: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_145, [8, 196, 768]);  mm_145 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_449, [1, 0]);  permute_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_78: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_15, unsqueeze_11);  add_15 = None
    div_79: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_78, unsqueeze_11);  div_78 = None
    neg_16: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_546)
    mul_467: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_16, div_79);  neg_16 = div_79 = None
    div_80: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_546, unsqueeze_11);  view_546 = unsqueeze_11 = None
    sum_205: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [3], True);  mul_467 = None
    squeeze_8: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_205, -1);  sum_205 = None
    unsqueeze_68: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_8, -1);  squeeze_8 = None
    expand_97: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_68, [8, 16, 196, 196]);  unsqueeze_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_192: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_80, expand_97);  div_80 = expand_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_468: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, sigmoid_2)
    mul_469: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, div_4)
    sum_206: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 2, 3], True);  mul_469 = None
    sub_167: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
    mul_470: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_167);  sigmoid_2 = sub_167 = None
    mul_471: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_206, mul_470);  sum_206 = None
    mul_472: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, sub_10);  sub_10 = None
    mul_473: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_192, div_3);  add_192 = None
    sum_207: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3], True);  mul_473 = None
    neg_17: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_207);  sum_207 = None
    mul_475: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_17, mul_470);  neg_17 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_193: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_471, mul_475);  mul_471 = mul_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_550: "f32[16]" = torch.ops.aten.reshape.default(add_193, [16]);  add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_476: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_468, div_4);  mul_468 = None
    sum_208: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
    mul_477: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_4, sum_208);  div_4 = sum_208 = None
    sub_169: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_478: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_472, div_3);  mul_472 = None
    sum_209: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
    mul_479: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_3, sum_209);  div_3 = sum_209 = None
    sub_170: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_480: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_170, 0.14433756729740643);  sub_170 = None
    view_551: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_480, [128, 196, 196]);  mul_480 = None
    bmm_66: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_452, view_551);  permute_452 = None
    bmm_67: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_551, permute_453);  view_551 = permute_453 = None
    view_552: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_66, [8, 16, 48, 196]);  bmm_66 = None
    view_553: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_67, [8, 16, 196, 48]);  bmm_67 = None
    permute_454: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_552, [0, 1, 3, 2]);  view_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_455: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_169, [0, 2, 3, 1]);  sub_169 = None
    sum_210: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_455, [0, 1, 2], True)
    view_554: "f32[16]" = torch.ops.aten.reshape.default(sum_210, [16]);  sum_210 = None
    clone_224: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None
    view_555: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_224, [307328, 16]);  clone_224 = None
    permute_456: "f32[16, 307328]" = torch.ops.aten.permute.default(view_555, [1, 0]);  view_555 = None
    mm_146: "f32[16, 3]" = torch.ops.aten.mm.default(permute_456, view_8);  permute_456 = None
    permute_457: "f32[3, 16]" = torch.ops.aten.permute.default(mm_146, [1, 0]);  mm_146 = None
    permute_458: "f32[16, 3]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_47: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_454, 0, 1);  permute_454 = None
    select_scatter_48: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_553, 0, 0);  view_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_194: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_47, select_scatter_48);  select_scatter_47 = select_scatter_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_459: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_194, [1, 3, 0, 2, 4]);  add_194 = None
    clone_225: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_459, memory_format = torch.contiguous_format);  permute_459 = None
    view_556: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_225, [8, 196, 1536]);  clone_225 = None
    view_557: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_556, [1568, 1536]);  view_556 = None
    permute_460: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_557, [1, 0])
    mm_147: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_460, view_31);  permute_460 = view_31 = None
    permute_461: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    mm_148: "f32[1568, 768]" = torch.ops.aten.mm.default(view_557, permute_462);  view_557 = permute_462 = None
    view_558: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_148, [8, 196, 768]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_195: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_549, view_558);  view_549 = view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_463: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_482: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_195, primals_8);  primals_8 = None
    mul_483: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, 768)
    sum_211: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
    mul_484: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_482, mul_10);  mul_482 = None
    sum_212: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
    mul_485: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_10, sum_212);  sum_212 = None
    sub_172: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_483, sum_211);  mul_483 = sum_211 = None
    sub_173: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_172, mul_485);  sub_172 = mul_485 = None
    mul_486: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_81, sub_173);  div_81 = sub_173 = None
    mul_487: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_195, mul_10);  mul_10 = None
    sum_213: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
    sum_214: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_196: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_191, mul_486);  add_191 = mul_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_559: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_196, [1568, 768])
    mm_149: "f32[1568, 3072]" = torch.ops.aten.mm.default(view_559, permute_464);  permute_464 = None
    permute_465: "f32[768, 1568]" = torch.ops.aten.permute.default(view_559, [1, 0])
    mm_150: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_465, view_25);  permute_465 = view_25 = None
    permute_466: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_150, [1, 0]);  mm_150 = None
    sum_215: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
    view_560: "f32[768]" = torch.ops.aten.reshape.default(sum_215, [768]);  sum_215 = None
    permute_467: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_466, [1, 0]);  permute_466 = None
    view_561: "f32[8, 196, 3072]" = torch.ops.aten.reshape.default(mm_149, [8, 196, 3072]);  mm_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_489: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
    mul_490: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, view_24)
    mul_491: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(mul_490, -0.5);  mul_490 = None
    exp_33: "f32[8, 196, 3072]" = torch.ops.aten.exp.default(mul_491);  mul_491 = None
    mul_492: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
    mul_493: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_24, mul_492);  view_24 = mul_492 = None
    add_198: "f32[8, 196, 3072]" = torch.ops.aten.add.Tensor(mul_489, mul_493);  mul_489 = mul_493 = None
    mul_494: "f32[8, 196, 3072]" = torch.ops.aten.mul.Tensor(view_561, add_198);  view_561 = add_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_562: "f32[1568, 3072]" = torch.ops.aten.reshape.default(mul_494, [1568, 3072]);  mul_494 = None
    mm_151: "f32[1568, 768]" = torch.ops.aten.mm.default(view_562, permute_468);  permute_468 = None
    permute_469: "f32[3072, 1568]" = torch.ops.aten.permute.default(view_562, [1, 0])
    mm_152: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_469, view_23);  permute_469 = view_23 = None
    permute_470: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_152, [1, 0]);  mm_152 = None
    sum_216: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
    view_563: "f32[3072]" = torch.ops.aten.reshape.default(sum_216, [3072]);  sum_216 = None
    permute_471: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_470, [1, 0]);  permute_470 = None
    view_564: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_151, [8, 196, 768]);  mm_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_496: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_564, primals_6);  primals_6 = None
    mul_497: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_496, 768)
    sum_217: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_496, [2], True)
    mul_498: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_496, mul_5);  mul_496 = None
    sum_218: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True);  mul_498 = None
    mul_499: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_5, sum_218);  sum_218 = None
    sub_175: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_497, sum_217);  mul_497 = sum_217 = None
    sub_176: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_175, mul_499);  sub_175 = mul_499 = None
    mul_500: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_82, sub_176);  div_82 = sub_176 = None
    mul_501: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(view_564, mul_5);  mul_5 = None
    sum_219: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 1]);  mul_501 = None
    sum_220: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_564, [0, 1]);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_199: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_196, mul_500);  add_196 = mul_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:76, code: x = self.proj(x)
    view_565: "f32[1568, 768]" = torch.ops.aten.reshape.default(add_199, [1568, 768])
    mm_153: "f32[1568, 768]" = torch.ops.aten.mm.default(view_565, permute_472);  permute_472 = None
    permute_473: "f32[768, 1568]" = torch.ops.aten.permute.default(view_565, [1, 0])
    mm_154: "f32[768, 768]" = torch.ops.aten.mm.default(permute_473, view_21);  permute_473 = view_21 = None
    permute_474: "f32[768, 768]" = torch.ops.aten.permute.default(mm_154, [1, 0]);  mm_154 = None
    sum_221: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_565, [0], True);  view_565 = None
    view_566: "f32[768]" = torch.ops.aten.reshape.default(sum_221, [768]);  sum_221 = None
    permute_475: "f32[768, 768]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
    view_567: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_153, [8, 196, 768]);  mm_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:75, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    view_568: "f32[8, 196, 16, 48]" = torch.ops.aten.reshape.default(view_567, [8, 196, 16, 48]);  view_567 = None
    permute_476: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    clone_228: "f32[8, 16, 196, 48]" = torch.ops.aten.clone.default(permute_476, memory_format = torch.contiguous_format);  permute_476 = None
    view_569: "f32[128, 196, 48]" = torch.ops.aten.reshape.default(clone_228, [128, 196, 48]);  clone_228 = None
    bmm_68: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(permute_477, view_569);  permute_477 = None
    bmm_69: "f32[128, 196, 196]" = torch.ops.aten.bmm.default(view_569, permute_478);  view_569 = permute_478 = None
    view_570: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_68, [8, 16, 196, 48]);  bmm_68 = None
    view_571: "f32[8, 16, 196, 196]" = torch.ops.aten.reshape.default(bmm_69, [8, 16, 196, 196]);  bmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:74, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_479: "f32[8, 196, 16, 48]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    clone_229: "f32[8, 196, 16, 48]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
    view_572: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(clone_229, [8, 196, 768]);  clone_229 = None
    view_573: "f32[1568, 768]" = torch.ops.aten.reshape.default(view_572, [1568, 768]);  view_572 = None
    permute_480: "f32[768, 1568]" = torch.ops.aten.permute.default(view_573, [1, 0])
    mm_155: "f32[768, 768]" = torch.ops.aten.mm.default(permute_480, view_5);  permute_480 = None
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    mm_156: "f32[1568, 768]" = torch.ops.aten.mm.default(view_573, permute_482);  view_573 = permute_482 = None
    view_574: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_156, [8, 196, 768]);  mm_156 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    div_83: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(add_5, unsqueeze_5);  add_5 = None
    div_84: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(div_83, unsqueeze_5);  div_83 = None
    neg_18: "f32[8, 16, 196, 196]" = torch.ops.aten.neg.default(view_571)
    mul_502: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(neg_18, div_84);  neg_18 = div_84 = None
    div_85: "f32[8, 16, 196, 196]" = torch.ops.aten.div.Tensor(view_571, unsqueeze_5);  view_571 = unsqueeze_5 = None
    sum_222: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [3], True);  mul_502 = None
    squeeze_9: "f32[8, 16, 196]" = torch.ops.aten.squeeze.dim(sum_222, -1);  sum_222 = None
    unsqueeze_69: "f32[8, 16, 196, 1]" = torch.ops.aten.unsqueeze.default(squeeze_9, -1);  squeeze_9 = None
    expand_98: "f32[8, 16, 196, 196]" = torch.ops.aten.expand.default(unsqueeze_69, [8, 16, 196, 196]);  unsqueeze_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:92, code: attn /= attn.sum(dim=-1).unsqueeze(-1)
    add_200: "f32[8, 16, 196, 196]" = torch.ops.aten.add.Tensor(div_85, expand_98);  div_85 = expand_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    mul_503: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, sigmoid)
    mul_504: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, div_1)
    sum_223: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 2, 3], True);  mul_504 = None
    sub_177: "f32[1, 16, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
    mul_505: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_177);  sigmoid = sub_177 = None
    mul_506: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(sum_223, mul_505);  sum_223 = None
    mul_507: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, sub_4);  sub_4 = None
    mul_508: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(add_200, div);  add_200 = None
    sum_224: "f32[1, 16, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3], True);  mul_508 = None
    neg_19: "f32[1, 16, 1, 1]" = torch.ops.aten.neg.default(sum_224);  sum_224 = None
    mul_510: "f32[1, 16, 1, 1]" = torch.ops.aten.mul.Tensor(neg_19, mul_505);  neg_19 = mul_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:91, code: attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
    add_201: "f32[1, 16, 1, 1]" = torch.ops.aten.add.Tensor(mul_506, mul_510);  mul_506 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:90, code: gating = self.gating_param.view(1, -1, 1, 1)
    view_575: "f32[16]" = torch.ops.aten.reshape.default(add_201, [16]);  add_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:88, code: pos_score = pos_score.softmax(dim=-1)
    mul_511: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_503, div_1);  mul_503 = None
    sum_225: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [-1], True)
    mul_512: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div_1, sum_225);  div_1 = sum_225 = None
    sub_179: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:87, code: patch_score = patch_score.softmax(dim=-1)
    mul_513: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(mul_507, div);  mul_507 = None
    sum_226: "f32[8, 16, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [-1], True)
    mul_514: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(div, sum_226);  div = sum_226 = None
    sub_180: "f32[8, 16, 196, 196]" = torch.ops.aten.sub.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:86, code: patch_score = (q @ k.transpose(-2, -1)) * self.scale
    mul_515: "f32[8, 16, 196, 196]" = torch.ops.aten.mul.Tensor(sub_180, 0.14433756729740643);  sub_180 = None
    view_576: "f32[128, 196, 196]" = torch.ops.aten.reshape.default(mul_515, [128, 196, 196]);  mul_515 = None
    bmm_70: "f32[128, 48, 196]" = torch.ops.aten.bmm.default(permute_484, view_576);  permute_484 = None
    bmm_71: "f32[128, 196, 48]" = torch.ops.aten.bmm.default(view_576, permute_485);  view_576 = permute_485 = None
    view_577: "f32[8, 16, 48, 196]" = torch.ops.aten.reshape.default(bmm_70, [8, 16, 48, 196]);  bmm_70 = None
    view_578: "f32[8, 16, 196, 48]" = torch.ops.aten.reshape.default(bmm_71, [8, 16, 196, 48]);  bmm_71 = None
    permute_486: "f32[8, 16, 196, 48]" = torch.ops.aten.permute.default(view_577, [0, 1, 3, 2]);  view_577 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:85, code: pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
    permute_487: "f32[8, 196, 196, 16]" = torch.ops.aten.permute.default(sub_179, [0, 2, 3, 1]);  sub_179 = None
    sum_227: "f32[1, 1, 1, 16]" = torch.ops.aten.sum.dim_IntList(permute_487, [0, 1, 2], True)
    view_579: "f32[16]" = torch.ops.aten.reshape.default(sum_227, [16]);  sum_227 = None
    clone_230: "f32[8, 196, 196, 16]" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
    view_580: "f32[307328, 16]" = torch.ops.aten.reshape.default(clone_230, [307328, 16]);  clone_230 = None
    permute_488: "f32[16, 307328]" = torch.ops.aten.permute.default(view_580, [1, 0]);  view_580 = None
    mm_157: "f32[16, 3]" = torch.ops.aten.mm.default(permute_488, view_8);  permute_488 = view_8 = None
    permute_489: "f32[3, 16]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    permute_490: "f32[16, 3]" = torch.ops.aten.permute.default(permute_489, [1, 0]);  permute_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    select_scatter_49: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, permute_486, 0, 1);  permute_486 = None
    select_scatter_50: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.select_scatter.default(full_default_2, view_578, 0, 0);  full_default_2 = view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:83, code: q, k = qk[0], qk[1]
    add_202: "f32[2, 8, 16, 196, 48]" = torch.ops.aten.add.Tensor(select_scatter_49, select_scatter_50);  select_scatter_49 = select_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_491: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.permute.default(add_202, [1, 3, 0, 2, 4]);  add_202 = None
    clone_231: "f32[8, 196, 2, 16, 48]" = torch.ops.aten.clone.default(permute_491, memory_format = torch.contiguous_format);  permute_491 = None
    view_581: "f32[8, 196, 1536]" = torch.ops.aten.reshape.default(clone_231, [8, 196, 1536]);  clone_231 = None
    view_582: "f32[1568, 1536]" = torch.ops.aten.reshape.default(view_581, [1568, 1536]);  view_581 = None
    permute_492: "f32[1536, 1568]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_158: "f32[1536, 768]" = torch.ops.aten.mm.default(permute_492, view_5);  permute_492 = view_5 = None
    permute_493: "f32[768, 1536]" = torch.ops.aten.permute.default(mm_158, [1, 0]);  mm_158 = None
    mm_159: "f32[1568, 768]" = torch.ops.aten.mm.default(view_582, permute_494);  view_582 = permute_494 = None
    view_583: "f32[8, 196, 768]" = torch.ops.aten.reshape.default(mm_159, [8, 196, 768]);  mm_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    add_203: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(view_574, view_583);  view_574 = view_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:82, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_495: "f32[1536, 768]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_517: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_203, primals_3);  primals_3 = None
    mul_518: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_517, 768)
    sum_228: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_517, [2], True)
    mul_519: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul_517, mul);  mul_517 = None
    sum_229: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_519, [2], True);  mul_519 = None
    mul_520: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(mul, sum_229);  sum_229 = None
    sub_182: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(mul_518, sum_228);  mul_518 = sum_228 = None
    sub_183: "f32[8, 196, 768]" = torch.ops.aten.sub.Tensor(sub_182, mul_520);  sub_182 = mul_520 = None
    mul_521: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(div_86, sub_183);  div_86 = sub_183 = None
    mul_522: "f32[8, 196, 768]" = torch.ops.aten.mul.Tensor(add_203, mul);  mul = None
    sum_230: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 1]);  mul_522 = None
    sum_231: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_204: "f32[8, 196, 768]" = torch.ops.aten.add.Tensor(add_199, mul_521);  add_199 = mul_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:364, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    sum_232: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_332, [0], True);  slice_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/convit.py:362, code: x = x + self.pos_embed
    sum_233: "f32[1, 196, 768]" = torch.ops.aten.sum.dim_IntList(add_204, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_496: "f32[8, 768, 196]" = torch.ops.aten.permute.default(add_204, [0, 2, 1]);  add_204 = None
    view_584: "f32[8, 768, 14, 14]" = torch.ops.aten.reshape.default(permute_496, [8, 768, 14, 14]);  permute_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_584, primals_181, primals_63, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_584 = primals_181 = primals_63 = None
    getitem_57: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_58: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return [sum_233, sum_232, sum_230, sum_231, view_575, sum_219, sum_220, sum_213, sum_214, view_550, sum_202, sum_203, sum_196, sum_197, view_525, sum_185, sum_186, sum_179, sum_180, view_500, sum_168, sum_169, sum_162, sum_163, view_475, sum_151, sum_152, sum_145, sum_146, view_450, sum_134, sum_135, sum_128, sum_129, view_425, sum_117, sum_118, sum_111, sum_112, view_400, sum_100, sum_101, sum_94, sum_95, view_375, sum_83, sum_84, sum_77, sum_78, view_350, sum_66, sum_67, sum_60, sum_61, sum_54, sum_55, sum_48, sum_49, sum_42, sum_43, sum_36, sum_37, getitem_57, getitem_58, permute_495, permute_490, view_579, permute_483, permute_475, view_566, permute_471, view_563, permute_467, view_560, permute_463, permute_458, view_554, permute_451, permute_443, view_541, permute_439, view_538, permute_435, view_535, permute_431, permute_426, view_529, permute_419, permute_411, view_516, permute_407, view_513, permute_403, view_510, permute_399, permute_394, view_504, permute_387, permute_379, view_491, permute_375, view_488, permute_371, view_485, permute_367, permute_362, view_479, permute_355, permute_347, view_466, permute_343, view_463, permute_339, view_460, permute_335, permute_330, view_454, permute_323, permute_315, view_441, permute_311, view_438, permute_307, view_435, permute_303, permute_298, view_429, permute_291, permute_283, view_416, permute_279, view_413, permute_275, view_410, permute_271, permute_266, view_404, permute_259, permute_251, view_391, permute_247, view_388, permute_243, view_385, permute_239, permute_234, view_379, permute_227, permute_219, view_366, permute_215, view_363, permute_211, view_360, permute_207, permute_202, view_354, permute_195, permute_187, view_341, permute_183, view_338, permute_179, view_335, permute_175, permute_164, view_321, permute_160, view_318, permute_156, view_315, permute_152, permute_141, view_301, permute_137, view_298, permute_133, view_295, permute_129, view_293, None]
    